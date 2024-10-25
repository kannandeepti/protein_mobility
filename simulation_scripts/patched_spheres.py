""" 
Simulation of multivalent spheres in 3D
=======================================

Uses the package polychrom to run molecular dynamics simulations of 
multivalent spherical "proteins". Each protein consists of a large sphere with
sticky patches on its surface, which represent cysteins that can participate in disulfide bonds.
The large spheres are self-avoiding, and disulfide bonds are modeled via an attractive
potential between the sticky patches.
 

Deepti Kannan, 2023
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from itertools import product
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
                         patch_attr_radius=0.2,
                         rep_radius=1.2,
                         rep_energy=50.0,
                         timestep=1,
                         collision_rate=10.0,
                         nblocks=1000,
                         blocksize=2000,
                         resume=False,
                         PBCbox=True,
                         **kwargs):
    """

    Parameters
    ----------
    N : int
        number of spheres
    f : int
        valency (number of patches per sphere)
    volume_fraction : float
        volume fraction of spheres in box with periodic boundary conditions
    E0 : float
        patch-patch attraction energy (kBT)
    savepath : str
        directory in which to store simulation results
    patch_attr_radius : float
        distance below which patches attract one another
    rep_radius : float
        distance at which large spheres repel each other
    rep_energy : float
        height of repulsive potential (kBT)
    timestep : int
        simulation time step (femtoseconds)
    collision_rate : float
        parameter for Brownian dynamics integrator (inverse picoseconds)
    nblocks : int
        run `nblocks` blocks of `blocksize` timesteps. save data every block.
    blocksize : int
        number of time steps in a block. save data every block.
    resume : bool
        whether to continue simulation from last available time step.
    PBCbox : bool
        whether to use periodic boundary conditions
    

    Returns
    -------
    ran_sim : bool
        whether simulation ran successfully

    """
    if 'L' in kwargs:
        L = kwargs.get('L')
    else:
        L = ((N * (4/3) * np.pi * ((rep_radius-0.05)/2)**3) / volume_fraction) ** (1/3)
    r = ((N * ((rep_radius-0.05)/2)**3) / volume_fraction) ** (1 / 3)
    print(f"PBC box size = {L}")
    ran_sim = False
    savepath = Path(savepath)
    if not savepath.is_dir():
        savepath.mkdir(parents=True)
    if patch_attr_radius != 0.5:
        savepath = savepath/f"N{N}_f0_2_E0{E0}_v{volume_fraction}_r{patch_attr_radius}_rep{rep_radius}_Erep{rep_energy}_dt{timestep}"
    else:
        if rep_radius==1.05:
            savepath = savepath/f"N{N}_f{f}_E0{E0}_v{volume_fraction}"
        elif PBCbox:
            savepath = savepath/f"N{N}_f{f}_E0{E0}_v{volume_fraction}_rep{rep_radius}"
        else:
            savepath = savepath/f"N{N}_f{f}_E0{E0}_v{volume_fraction}_rep{rep_radius}_conf"
    
    if savepath.is_dir() and not resume:
        print("Simulation has already been run. Exiting.")
        return ran_sim
    
    if resume:
        save_folder = savepath/'resume'
    else:
        save_folder = savepath
    
    if PBCbox:
        PBCbox = (L, L, L)

    reporter = HDF5Reporter(folder=save_folder, 
            max_data_length=1000, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="brownian",
        error_tol=0.003,
        GPU=f"{gpuid}",
        collision_rate=collision_rate,
        N=N*(f+1),
        save_decimals=2,
        timestep=timestep,
        PBCbox=PBCbox,
        reporters=[reporter],
    )
    if resume:
        data_so_far = list_URIs(savepath)
        starting_pos = load_URI(data_so_far[-1])["pos"]
    else:
        #choose initial positions of spheres by doing a local energy minimization
        #based on all forces except the patch-patch attraction
        starting_pos = energy_minimization(gpuid, N, f, volume_fraction, 
                         timestep,
                         PBCbox,
                         reporter,
                         rep_radius,
                         rep_energy)
        
    sim.set_data(starting_pos, center=True)
    sim.set_velocities(v=np.zeros(((f+1)*N, 3)))
    if not PBCbox:
        sim.add_force(forces.spherical_confinement(sim, r=r, k=5.0))
    sim.add_force(patched_forces.patched_particle_forcekit(
        sim,
        N,
        f,
        #make every other particle sticky
        sticky_subset=np.arange(0, N, 2),
        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            'bondLength' : 0.5,
            'bondWiggleDistance' : 0.01,
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
            'trunc' : rep_energy,
            'radiusMult' : rep_radius
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

def energy_minimization(gpuid, N, f, volume_fraction,
                         timestep,
                         PBCbox,
                         reporter,
                         rep_radius,
                         rep_energy): 
    
    """ Create a simulation object to perform a local energy minimization 
    based on all forces except for the patch-patch attraction. """  

    r = ((N * ((rep_radius-0.05)/2)**3) / volume_fraction) ** (1 / 3)
    print(f'Grow cubic box size = {2*r}')

    minimizer = simulation.Simulation(
        platform="CUDA",
        integrator="brownian",
        error_tol=0.003,
        GPU=f"{gpuid}",
        collision_rate=5.0,
        N=N*(f+1),
        save_decimals=2,
        timestep=timestep,
        PBCbox=PBCbox,
        reporters=[reporter],
    )
    if N > 2:
        positions = starting_conformations.grow_cubic(N, int(2*r))
    else:
        positions = np.array([[0., 0., 0.]])
    patch_points = patched_particle_geom(f, R=0.5)
    starting_pos = [atom_pos.reshape((1, 3)) + patch_points for atom_pos in positions]
    starting_pos = np.array(starting_pos).reshape(((f+1)*N, 3)) 
    minimizer.set_data(starting_pos, center=True)
    minimizer.set_velocities(v=np.zeros(((f+1)*N, 3)))
    
    if not PBCbox:
        minimizer.add_force(forces.spherical_confinement(sim, r=r, k=5.0))
    minimizer.add_force(patched_forces.patched_particle_forcekit(
        minimizer,
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
        patch_attraction_force_func=None,
        nonbonded_force_func=patched_forces.patched_particle_repulsive,
        nonbonded_force_kwargs={
            'trunc' : rep_energy,
            'radiusMult' : rep_radius
        },
        exclude_intramolecular=True,
        #patches anyway not in interaciton group, so dont need to create exclusions from bonds
        except_bonds=False
    ))
    minimizer.local_energy_minimization()
    pts = minimizer.get_data()
    return pts

def batch_tasks(E0_values, f_values, rep_radii, attr_radii, rep_energies, timesteps,
                gpuid=0, N=1000, vol_fraction=0.3,
                **kwargs):

    """ Distribute parameter sweep over multiple cores. """
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # parameters to sweep
    params_to_sweep = list(product(f_values, E0_values, rep_radii, attr_radii, rep_energies, timesteps))
    # batch to process with this task
    params_per_task = params_to_sweep[my_task_id: len(params_to_sweep): num_tasks]
    print(params_per_task)
    tic = time.time()
    sims_ran = 0
    blocksize_dt10 = 20000
    for param_set in params_per_task:
        f, E0, rep_r, attr_r, Erep, dt = param_set
        nblocks_dt10 = 5000
        #if f==2:
        #    nblocks_dt10 = 10000
        time_factor = 10 / dt
        blocksize = blocksize_dt10 * time_factor
        print(f"Running simulation with N={N}, f={f}, E0={E0}, repr={rep_r}, attr={attr_r}, Erep={Erep}, dt={dt}")
        ran_sim = simulate_patched_spheres(gpuid, N, f, vol_fraction, E0, "results", rep_radius=rep_r,
                                           patch_attr_radius=attr_r, rep_energy=Erep, timestep=dt,
                                           nblocks=nblocks_dt10, blocksize=blocksize, 
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
    E0_values = [11.70, 12.20, 12.97, 13.5, 14.1, 14.7, 15.65, 16.8, 17.3, 17.8, 
                 18.2, 18.6, 18.9, 19.25, 19.6, 19.9, 20.23, 20.48, 20.69, 20.84,
                 20.95, 21.015, 21.02, 21.028]
    f_values = [2]
    rep_radii = [1.2]
    attr_radii = [0.2]
    rep_energies = [50.0]
    timesteps = [2.5]
    batch_tasks(E0_values, f_values, rep_radii, attr_radii, rep_energies, timesteps,
            vol_fraction=0.3, N=1000, PBCbox=True, resume=True)

