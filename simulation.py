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
                         savepath,
                         collision_rate=0.03,
                         nblocks=1000,
                         blocksize=100,
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

    Returns
    -------

    """
    L = ((N * (4/3) * np.pi * (0.5)**3) / volume_fraction) ** (1/3)
    print(L)
    reporter = HDF5Reporter(folder=savepath, max_data_length=100, overwrite=True)
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
            'attractionEnergy' : 3.0,
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

def evolve_one_generation(sim, bondStepper, time):
    """ Integrate for some amount of time and then update bonds."""
    curtime = sim.state.getTime() / simtk.unit.picosecond
    sim.integrator.stepTo(curtime + time)
    curBonds, pastBonds = bondStepper.step(curtime, sim.context)

if __name__ == "__main__":
    tic = time.time()
    simulate_patched_spheres(0, 100, 3, 0.3, "test_patch_attr")
    toc = time.time()
    nsecs = toc - tic

    print(f"Ran simulation in {nsecs}s")
