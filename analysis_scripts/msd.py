r"""
Computing MSDS from polychrom simulations
=========================================

Script to calculate mean squared displacements over time
from output of polychrom simulations. MSDs can either be computed
by (1) averaging over an ensemble of trajectories or (2) time lag averaging
using a single trajectory.

Deepti Kannan. 2023
"""

import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from numba import jit
from polychrom.hdf5_format import list_URIs, load_hdf5_file, load_URI

def extract_particle_trajectory(simdir, f, N=1000, start=5000, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir
    and store particle positions over time in a matrix X.

    Parameters
    ----------
    f : int
        valency
    E0 : float
        patch-patch attraction energy
    simdir : str or Path
        path to parent directory containing simulation folders named "N{N}_f{f}_E0{E0}_v{v}"
    start : int
        which time block to start loading conformations from
    every_other : int
        skip every_other time steps when loading conformations

    Returns
    -------
    X : array_like (num_t, num_particles, 3)
        x, y, z positions of all particles (excluding patches) over time

    """
    molecule_inds = np.arange(0, (f+1)*N, f+1).astype(int)
    totalN = (f + 1) * N
    X = []
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(Path(simdir) / "starting_conformation_0.h5")[
            "pos"
        ]
        X.append(starting_pos)
    for conformation in data[start::every_other]:
        pos = load_URI(conformation)["pos"]
        ncopies = pos.shape[0] // totalN
        for i in range(ncopies):
            posN = pos[totalN * i : totalN * (i + 1)]
            X.append(posN)
    X = np.array(X)
    Xparticle = X[:, molecule_inds, :]
    return Xparticle


@jit(nopython=True)
def get_bead_msd_time_ave(X):
    """Calculate time lag averaged MSDs of particles in a single simulation trajectory stored in X.

    Parameters
    ----------
    X : np.ndarray (num_t, num_particles, d)
        trajectory of particle positions in d dimensions over num_t timepoints
        
    Returns
    -------
    msd_ave : (num_t - 1,)
        time lag averaged MSD averaged over all particles

    """
    num_t, num_particles, d = X.shape
    msd = np.zeros((num_t - 1,))
    count = np.zeros((num_t - 1,))
    for i in range(num_t - 1):
        for j in range(i, num_t - 1):
            diff = X[j] - X[i]
            msd[j - i] += np.mean(np.sum(diff * diff, axis=-1))
            count[j - i] += 1
    msd_ave = msd / count
    return msd_ave

def save_time_ave_msd(f, E0, r=0.5, repr=1.05, PBCbox=False,
                      start=5000*100, every_other=10):
    """ TODO: time units are wrong. Missing a factor of 2000 steps per block."""
    if f < 4:
        time_per_step = 100
    else:
        time_per_step = 70
    start = int(np.floor(start / time_per_step))
    if r != 0.5:
        path = Path(f"results/N1000_f{f}_E0{E0}_v0.3_r{r}_rep{repr}")
    elif repr == 1.05:
        path = Path(f"results/N1000_f{f}_E0{E0}_v0.3")
    elif PBCbox:
        path = Path(f"results/N1000_f{f}_E0{E0}_v0.3_rep{repr}")
    else:
        path = Path(f"results/N1000_f{f}_E0{E0}_v0.3_rep{repr}_conf") 

    msdfile = path/f'time_ave_msd_every_other_10_start{start}.csv'
    if path.is_dir() and not msdfile.is_file():
        X = extract_particle_trajectory(path, f, start=start, every_other=10)
        if (path/"resume").is_dir():
            Y = extract_particle_trajectory(path/"resume", f, start=1, every_other=10)
            X = np.concatenate((X, Y), axis=0)
        msd = get_bead_msd_time_ave(X)
        df = pd.DataFrame()
        #in units of femtoseconds (arbirtary unit used by openMM)
        df['Time'] = np.arange(0, len(msd)) * every_other * time_per_step
        df['MSD'] = msd
        df.to_csv(msdfile, index=False)

if __name__ == "__main__":
    f_values = [1, 2]
    start_values = [5000*100]
    E0_values = [0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0, 20.0, 25.0, 30.0]
    rep_radii = [1.05, 1.1]
    attr_radii = [0.35]
    
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # parameters to sweep
    params_to_sweep = []
    for E0 in E0_values:
        for start in start_values:
            for f in f_values:
                for rep_r in rep_radii:
                    for attr_r in attr_radii:
                        params_to_sweep.append((f, E0, start, rep_r, attr_r))
    # batch to process with this task
    params_per_task = params_to_sweep[my_task_id: len(params_to_sweep): num_tasks]
    print(params_per_task)
    tic = time.time()
    for param_set in params_per_task:
        f, E0, start, rep_r, attr_r = param_set
        save_time_ave_msd(f, E0, r=attr_r, repr=rep_r, start=start, PBCbox=True)
    toc = time.time()
    print(f"Ran {len(params_per_task)} MSD calculations in {toc - tic}sec")
