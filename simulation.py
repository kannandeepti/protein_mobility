""" Simulation of WCA spheres in 3D. """

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

def polynomial_repulsive(sim_object, particles=None, trunc=5.0, radiusMult=1.0,
                         name="polynomial_repulsive"):
    """This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    See the gist below with an example of the potential.
    https://gist.github.com/mimakaev/0327bf6ffe7057ee0e0625092ec8e318

    Parameters
    ----------

    trunc : float
        the energy value around r=0

    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    repul_energy = (
        "rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
    )

    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name

    force.addGlobalParameter("REPe", trunc * sim_object.kT)
    force.addGlobalParameter("REPsigma", radius)
    # Coefficients for x^8*(x*x-1)
    # force.addGlobalParameter('emin12', 256.0 / 3125.0)
    # force.addGlobalParameter('rmin12', 2.0 / np.sqrt(5.0))
    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    particles = range(sim_object.N) if particles is None else particles
    for i in particles:
        force.addParticle([])

    force.setCutoffDistance(nbCutOffDist)

    return force

def simulate_WCA_spheres(gpuid, N, volume_fraction,
                         savepath,
                         collision_rate=0.03, **kwargs):
    """

    Parameters
    ----------
    N : int
        number of spheres
    volume_fraction : float
        volume fraction of spheres in PBC box

    Returns
    -------

    """
    L = ((N * (0.5)**3) / volume_fraction) ** (1/3)
    reporter = HDF5Reporter(folder=savepath, max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="variableLangevin",
        error_tol=0.003,
        GPU=f"{gpuid}",
        collision_rate=0.03,
        N=N,
        save_decimals=2,
        PBCbox=(L, L, L),
        reporters=[reporter],
    )
    positions = starting_conformations.grow_cubic(N, int(2*L))
    sim.set_data(positions, center=True)
    sim.add_force(polynomial_repulsive(sim, trunc=5.0))

    for _ in range(10):
        sim.do_block(100)
    sim.print_stats()
    reporter.dump_data()

if __name__ == "__main__":
    simulate_WCA_spheres(0, 100, 0.3, "trajectory")


