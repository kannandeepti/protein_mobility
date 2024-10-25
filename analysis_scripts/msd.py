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
from itertools import product
import numpy as np
import pandas as pd
from numba import jit
from polychrom.hdf5_format import list_URIs, load_hdf5_file, load_URI

def extract_particle_trajectory(simdir, f, N=1000, start=5000, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir
    and store particle positions over time in a matrix X.

    Parameters
    ----------
    simdir : str or Path
        path to parent directory containing simulation folders named "N{N}_f{f}_E0{E0}_v{v}"
    f : int
        valency
    N : int
        number of spheres
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

def save_time_ave_msd(f, E0, N=1000, r=0.5, repr=1.05, v=0.3, Erep=30.0, PBCbox=False,
                      start=5000*50*2000, timestep=50,
                      blocksize=2000, every_other=10):
    """ Save time-averaged mean squared displacement of spheres over time for
    given parameter set.
    
    Parameters
    ----------
    f : int
        valency
    N : int
        number of spheres
    r : float
        patch-patch attraction radius
    repr : float
        repulsion radius of protein spheres
    v : float
        volume fraction of spheres in PBC box
    Erep : float
        repulsion energy (kbT)
    PBCbox : bool
        whether simulations were run with periodic boundary conditions
    start : int
        which time block to start loading conformations from
    timestep : int
        simulation timestep 
    blocksize : int
        number of timesteps in a simulation block
    every_other : int
        skip every_other time steps when loading conformations
    
    """
    
    if r != 0.5:
        path = Path(f"results/N{N}_f0_2_E0{E0}_v{v}_r{r}_rep{repr}_Erep{Erep}_dt{timestep}")
    elif repr == 1.05:
        path = Path(f"results/N{N}_f{f}_E0{E0}_v{v}")
    elif PBCbox:
        path = Path(f"results/N{N}_f{f}_E0{E0}_v{v}_rep{repr}")
    else:
        path = Path(f"results/N{N}_f{f}_E0{E0}_v{v}_rep{repr}_conf") 
    #read time step from initArgs file
    init_args = load_hdf5_file(path/'initArgs_0.h5')
    if (timestep != init_args['timestep']):
        raise ValueError('supplied time step does not match simulation parameter')
    timestep = init_args['timestep']
    start = int(np.floor(start / timestep / blocksize))
    msdfile = path/f'time_ave_msd_every_other_{every_other}_start{start}.csv'
    if path.is_dir():
        X = extract_particle_trajectory(path, f, N=N, start=start, every_other=every_other)
        if (path/"resume").is_dir():
            Y = extract_particle_trajectory(path/"resume", f, N=N, start=1, every_other=every_other)
            X = np.concatenate((X, Y), axis=0)
        #subset that has f=2
        sticky_subset = np.arange(0, N, 2)
        #subset that has f=0
        non_sticky_subset = np.setdiff1d(np.arange(0, N, 1), sticky_subset)
        msd_f2 = get_bead_msd_time_ave(X[:, sticky_subset, :])
        msd_f0 = get_bead_msd_time_ave(X[:, non_sticky_subset, :])
        df = pd.DataFrame()
        #in units of femtoseconds (arbirtary unit used by openMM)
        df['Time'] = np.arange(0, len(msd_f2)) * every_other * timestep * blocksize
        df['MSD_f2'] = msd_f2
        df['MSD_f0'] = msd_f0
        df.to_csv(msdfile, index=False)

if __name__ == "__main__":
    E0_values = [11.70, 12.20, 12.97, 13.5, 14.1, 14.7, 15.65, 16.8, 17.3, 17.8, 
                 18.2, 18.6, 18.9, 19.25, 19.6, 19.9, 20.23, 20.48, 20.69, 20.84,
                 20.95, 21.015, 21.02, 21.028]
    f_values = [2]
    rep_radii = [1.2]
    attr_radii = [0.2]
    rep_energies = [50.0]
    timesteps = [2.5]
    block_starts = [500]
    
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # batch to process with this task
    params_to_sweep = list(product(f_values, E0_values, rep_radii, attr_radii, rep_energies, timesteps, block_starts))
    params_per_task = params_to_sweep[my_task_id: len(params_to_sweep): num_tasks]
    print(params_per_task)
    blocksize_dt10 = 20000
    tic = time.time()
    for param_set in params_per_task:
        f, E0, rep_r, attr_r, Erep, dt, block_start = param_set
        nblocks_dt10 = 10000
        #if f == 1:
        #    nblocks_dt10 = 3000
        #else:
        #    nblocks_dt10 = 10000
        time_factor = 10 / dt
        blocksize = blocksize_dt10 * time_factor
        start = block_start * dt * blocksize
        save_time_ave_msd(f, E0, r=attr_r, repr=rep_r, Erep=Erep, v=0.3, start=start,
                          every_other=1, N=1000, 
                          timestep=dt, blocksize=blocksize, PBCbox=True)
    toc = time.time()
    print(f"Ran {len(params_per_task)} MSD calculations in {toc - tic}sec")
