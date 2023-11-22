""" Simulation of multivalent spheres in 3D. """

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from typing import Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import openmm
except Exception:
    import simtk.openmm as openmm

sys.path.append(os.getcwd())
import polychrom
from polychrom import simulation, forces, starting_conformations
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI

import utils.forces as patched_forces
from utils.geometry import patched_particle_geom


def simulate_patched_spheres(gpuid, N, f, volume_fraction,
                         E0,
                         savepath,
                         patch_attr_radius=0.5,
                         collision_rate=0.03,
                         nblocks=1000,
                         blocksize=2000,
                         resume=False,
                         **kwargs):
    """

    Parameters
    ----------
    N : int
        number of spheres
    f : int
        valency (number of patches per sphere)
    volume_fraction : float
        volume fraction of spheres in PBC box
    E0 : float
        patch-patch attraction energy (sets koff of bond)

    Returns
    -------

    """
    if 'L' in kwargs:
        L = kwargs.get('L')
    else:
        L = ((N * (4/3) * np.pi * (0.5)**3) / volume_fraction) ** (1/3)
    print(f"PBC box size = {L}")
    ran_sim = False
    savepath = Path(savepath)
    if not savepath.is_dir():
        savepath.mkdir(parents=True)
    if patch_attr_radius != 0.5:
        savepath = savepath/f"N{N}_f{f}_E0{E0}_v{volume_fraction}_r{patch_attr_radius}"
    else:
        savepath = savepath/f"N{N}_f{f}_E0{E0}_v{volume_fraction}"
    if savepath.is_dir() and not resume:
        print("Simulation has already been run. Exiting.")
        return ran_sim
    
    if resume:
        save_folder = savepath/'resume'
    else:
        save_folder = savepath

    if f <= 3:
        timestep = 100
    else:
        timestep = 70

    reporter = HDF5Reporter(folder=save_folder, 
            max_data_length=1000, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="brownian",
        error_tol=0.003,
        GPU=f"{gpuid}",
        collision_rate=5.0,
        N=N*(f+1),
        save_decimals=2,
        timestep=timestep,
        PBCbox=(L, L, L),
        reporters=[reporter],
    )
    if resume:
        data_so_far = list_URIs(savepath)
        starting_pos = load_URI(data_so_far[-1])["pos"]
    else:
        if N > 2:
            positions = starting_conformations.grow_cubic(N, int(2*L))
        else:
            positions = np.array([[0., 0., 0.]])
        patch_points = patched_particle_geom(f, R=0.5)
        print(patch_points.shape)
        starting_pos = [atom_pos.reshape((1, 3)) + patch_points for atom_pos in positions]
        starting_pos = np.array(starting_pos).reshape(((f+1)*N, 3))
    sim.set_data(starting_pos, center=True)
    sim.set_velocities(v=np.zeros(((f+1)*N, 3)))
    particle_inds = np.arange(0, (f + 1) * N, 1)
    #indices of larger spheres
    molecule_inds = np.arange(0, (f+1)*N, f+1)
    #indices of patches
    patch_inds = np.setdiff1d(particle_inds, molecule_inds)
    #sim.add_force(forces.spherical_confinement(sim, r=L, k=5.0))
    sim.add_force(patched_forces.patched_particle_forcekit(
        sim,
        N,
        f,
        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            'bondLength' : 0.5,
            'bondWiggleDistance' : 0.05,
        },
        angle_force_func=forces.angle_force,
        angle_force_kwargs={
            'k' : 30.0
        },
        dihedral_force_func=patched_forces.dihedral_force,
        dihedral_force_kwargs={
            'k' : 30.0
        },
        patch_attraction_force_func=patched_forces.patch_attraction,
        patch_attraction_force_kwargs={
            'attractionEnergy' : E0,
            'attractionRadius' : patch_attr_radius
        },
        nonbonded_force_func=patched_forces.patched_particle_repulsive,
        nonbonded_force_kwargs={
            'trunc' : 30.0,
            'radiusMult' : 1.05
        },
        exclude_intramolecular=True,
        #patches anyway not in interaciton group, so dont need to create exclusions from bonds
        except_bonds=False
    ))
    for _ in range(nblocks):
        sim.do_block(blocksize)
    sim.print_stats()
    reporter.dump_data()
    ran_sim = True
    return ran_sim

def evolve_one_generation(sim, bondStepper, time):
    """ Integrate for some amount of time and then update bonds."""
    curtime = sim.state.getTime() / simtk.unit.picosecond
    sim.integrator.stepTo(curtime + time)
    curBonds, pastBonds = bondStepper.step(curtime, sim.context)

def batch_tasks(E0_values, f_values, gpuid=0, N=1000, vol_fraction=0.3,
                **kwargs):
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # parameters to sweep
    params_to_sweep = [(0, 0.0)]
    for E0 in E0_values:
        for f in f_values:
            params_to_sweep.append((f, E0))
    # batch to process with this task
    params_per_task = params_to_sweep[my_task_id: len(params_to_sweep): num_tasks]
    print(params_per_task)
    tic = time.time()
    sims_ran = 0
    for param_set in params_per_task:
        f, E0 = param_set
        print(f"Running simulation with f={f}, E0={E0}")
        ran_sim = simulate_patched_spheres(gpuid, N, f, vol_fraction, E0, "results",
                                           **kwargs)
        if ran_sim:
            sims_ran += 1
    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s")

if __name__ == "__main__":
    E0_values = [0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 15.0]
    f_values = [1, 2]
    batch_tasks(E0_values, f_values, patch_attr_radius=0.25, nblocks=20000,
            blocksize=2000, resume=False)

