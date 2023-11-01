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
from polychrom.hdf5_format import HDF5Reporter

import utils.forces as patched_forces

def patched_particle_geom(f, R=1):
    """ Distribute f residues on a sphere with equal angles."""

    # first position is center particle
    positions = [[0., 0., 0.]]
    theta = np.pi / 2
    for i in range(min(f, 5)):
        # for valency less than 5, just distribute points on a circle in the x-y plane
        phi = 2 * np.pi * i / f
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        positions.append([x, y, z])

    if f >= 5:
        # octahedron -> put 5th particle perpendicular to plain of 4 points
        raise ValueError("Have not implemented yet")
    return np.array(positions)

def simulate_patched_spheres(gpuid, N, f, volume_fraction,
                         E0,
                         savepath,
                         collision_rate=0.03,
                         nblocks=1000,
                         blocksize=2000,
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
    L = ((N * (4/3) * np.pi * (0.5)**3) / volume_fraction) ** (1/3)
    print(f"PBC box size = {L}")
    ran_sim = False
    savepath=Path(savepath)/f"N{N}_f{f}_E0{E0}_v{volume_fraction}"
    if savepath.is_dir():
        print("Simulation has already been run. Exiting.")
        return ran_sim

    reporter = HDF5Reporter(folder=savepath, 
            max_data_length=1000, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="brownian",
        error_tol=0.003,
        GPU=f"{gpuid}",
        collision_rate=5.0,
        N=N*(f+1),
        save_decimals=2,
        timestep=100,
        PBCbox=(L, L, L),
        reporters=[reporter],
    )
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
        patch_attraction_force_func=patched_forces.patch_attraction,
        patch_attraction_force_kwargs={
            'attractionEnergy' : E0,
            'attractionRadius' : 0.5
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

if __name__ == "__main__":
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1]) 
    num_tasks = int(sys.argv[2])

    # parameters to sweep
    E0_values = [0.1, 0.2, 0.3, 0.4]
    f_values = [1, 2, 3]
    params_to_sweep = []
    for E0 in E0_values:
        for f in f_values:
            params_to_sweep.append((f, E0))
    # batch to process with this task
    params_per_task = params_to_sweep[my_task_id : len(params_to_sweep) : num_tasks]
    print(params_per_task)
    tic = time.time()
    sims_ran = 0
    for param_set in params_per_task:
        f, E0 = param_set
        print(f"Running simulation with f={f}, E0={E0}")
        ran_sim = simulate_patched_spheres(0, 1000, f, 0.3, E0, "results",
                nblocks=20000, blocksize=2000)
        if ran_sim:
            sims_ran += 1
    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s")

