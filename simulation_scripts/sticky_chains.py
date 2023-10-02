"""
Script to run hybrid simulations with active forces and sticky B-B attractions 

A/B identities inferred from q-arm of chr 2 in chromatin tracing data of Su et al. (2020).

Deepti Kannan, 2023
"""
import time
import numpy as np
import pandas as pd
import os, sys
from scipy.spatial.distance import pdist
sys.path.append(os.getcwd())
os.chdir("/home/dkannan/git-remotes/polychrom/")
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter
import openmm
from simtk import unit
from pathlib import Path

def sticky_inds(N, f, l):
    """ Return particle IDs of sticky patches on a chain of N monomers
    with f sticky patches of length l that are uniformly distributed along the chain.
    
    Assumes ends of the chain are not sticky.
    linker - sticky - linker

    Returns
    -------
    ids : (N,) array-like
        array of 0's and 1's where 1's coorespond to sticky patches
    """
    ids = np.zeros(N)
    nlinkers = f + 1
    #need at least f*l + nlinkers monomers to define a chain
    if (f * l + nlinkers) < N:
        raise ValueError("Need at least f(l+1) + 1 monomers in a chain")
    total_linker_length = N - f * l
    if (total_linker_length % nlinkers == 0):
        linkers = np.tile(int(total_linker_length / nlinkers), nlinkers)
    else:
        #number of short linkers of length np.floor(total_linker_length / nlinkers)
        nshort = nlinkers - (total_linker_length % nlinkers)
        #number of long linkers of length np.ceil(total_linker_length / nlinkers)
        nlong = nlinkers - nshort
        short_linkers = np.tile(int(np.floor(total_linker_length / nlinkers)), nshort)
        long_linkers = np.tile(int(np.ceil(total_linker_length / nlinkers)), nlong)
        linkers = np.concatenate((long_linkers, short_linkers))
    i = 0
    for link in linkers:
        sticker_start = i + link
        ids[sticker_start : sticker_start + l] = 1.0
        i += link + f
    return ids

def hcp(n):
    dim = 3
    k, j, i = [v.flatten()
               for v in np.meshgrid(*([range(n)] * dim), indexing='ij')]
    df = pd.DataFrame({
        'x': 2 * i + (j + k) % 2,
        'y': np.sqrt(3) * (j + 1/3 * (k % 2)),
        'z': 2 * np.sqrt(6) / 3 * k,
    })
    return df

def square_lattice(n):
    dim = 3
    k, j, i = [v.flatten()
               for v in np.meshgrid(*([range(n)] * dim), indexing='ij')]
    df = pd.DataFrame({
        'x': i,
        'y': j,
        'z': k,
    })
    return df

def initialize_territories(volume_fraction, mapN, nchains, lattice='hcp',
                          rs=None):
    r_chain = ((mapN * (0.5)**3) / volume_fraction) ** (1/3)
    r_confinement = ((nchains * mapN * (0.5)**3) / volume_fraction) ** (1/3)
    print(r_chain)
    print(r_confinement)
    #first calculate centroid positions of chains
    n_lattice_points = [i**3 for i in range(10)]
    lattice_size = np.searchsorted(n_lattice_points, nchains)
    print(f'Lattice size = {lattice_size}')
    if lattice=='hcp':
        df = hcp(lattice_size)
    if lattice=='square':
        df = square_lattice(lattice_size)
    else:
        raise ValueError('only hcp and square lattices implemented so far')
    df['radial_distance'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df.sort_values('radial_distance', inplace=True)
    positions = df.to_numpy()[:, :3][:nchains]
    if rs is None:
        #assume dense packing of spheres (rs is maximum allowed)
        # in units of sphere radii
        max_diameter = pdist(positions).max()
        #mini sphere size
        rs = np.floor(2*r_confinement / max_diameter)
        positions *= rs
    else:
        #rs is the desired cell size of each square in lattice
        positions *= rs
        #now set radius of sphere to be that of individual chain
        rs = r_chain
    starting_conf = []
    for i in range(nchains):
        centroid = positions[i]
        def confine_chrom(pos):
            x, y, z = pos
            #reject position if it's more than 5% outside of the spherical radius
            return ((np.sqrt((x - centroid[0])**2 + (y-centroid[1])**2 + (z-centroid[2])**2)) <= rs)
        chrom_pos = starting_conformations.create_constrained_random_walk(mapN, confine_chrom, starting_point=(centroid[0], centroid[1], centroid[2]))
        starting_conf.append(chrom_pos)
    starting_conf = np.array(starting_conf).reshape((nchains*mapN, 3))
    return starting_conf

def spherical_well_array(sim_object, r, cell_size, particles=None,
                         width=1, depth=1, name="spherical_well_array"):
    """
    An (array of) spherical potential wells. Uses floor functions to map
    particle positions to the coordinates of the well.

    Parameters
    ----------

    r : float
        Radius of the nucleus
    cell_size : float
        width of cell in lattice of spherical wells
    particles : list of int or np.array
        indices of particles that are attracted
    width : float, optional
        Width of attractive well, nm.
    depth : float, optional
        Depth of attractive potential in kT
        Positive means the walls are repulsive (i.e chain confined within lamina).
        Negative means walls are attractive (i.e. attraction to lamina)
    """

    force = openmm.CustomExternalForce(
        "step(1+d) * step(1-d) * SPHWELLdepth * (1 + cos(3.1415926536*d)) / 2;"
        "d = (sqrt((x1-SPHWELLx)^2 + (y1-SPHWELLy)^2 + (z1-SPHWELLz)^2) - SPHWELLradius) / SPHWELLwidth;"
        "x1 = x - L*floor(x/L);"
        "y1 = y - L*floor(y/L);"
        "z1 = z - L*floor(z/L);"
    )
    force.name = name
    particles = range(sim_object.N) if particles is None else particles
    center = 3 * [cell_size/2]
    
    force.addGlobalParameter("SPHWELLradius", r * sim_object.conlen)
    force.addGlobalParameter("SPHWELLwidth", width * sim_object.conlen)
    force.addGlobalParameter("SPHWELLdepth", depth * sim_object.kT)
    force.addGlobalParameter("L", cell_size * sim_object.conlen)
    force.addGlobalParameter("SPHWELLx", center[0] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLy", center[1] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLz", center[2] * sim_object.conlen)

    # adding all the particles on which force acts
    for i in particles:
        # NOTE: the explicit type cast seems to be necessary if we have an np.array...
        force.addParticle(int(i), [])

    return force
    
    
def run_sticky_sim(gpuid, run_number, N, nchains, E0, sticky_ids, volume_fraction=0.2,
                   width=10.0, depth=5.0, #spherical well array parameters
                   confine="PBCbox", timestep=170, nblocks=20000, blocksize=2000,
                   time_stepping_fn=None):
    """Run a single simulation on a GPU of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    run_number : int
        replicate number for this parameter set
    N : int
        number of monomers in each subchain
    nchains : int
        number of subchains in system
    E0 : float
        selective B-B attractive energy
    sticky_ids : (N,) array-like
        array of 0's and 1's where 1's indicate sticky particles in a given chain
    volume_fraction : float
        volume fraction of monomers within the confinement
    confine : str
        if "single", put all chains in a single spherical confinement with provided density/
        if "many", put each chain in its own spherical well where chains are arranged on a lattice.
        lattice spacing is 5*r, where r the radius of each mini sphere determined based on density.
        if "PBCbox" put all chains within a box with periodic boundary conditions.
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    ran_sim = False
    particle_inds = np.arange(0, N*nchains, dtype="int")
    sticky_inds = particle_inds[np.tile(sticky_ids, nchains)]
    # radius of spherical confinement for this volume fraction
    r_chain = ((N * (0.5)**3) / volume_fraction) ** (1/3)
    r = ((N * nchains * (0.5)**3) / volume_fraction) ** (1/3)
    #length of cube for PBC for this volume fraction
    L = ((N * nchains * (4/3) * np.pi * (0.5)**3) / volume_fraction)**(1/3)  
    print(f"Radius of confinement: {r}")
    print(f"Radius of confined chain: {r_chain}")
    print(f"Length of cubic box: {L}")
    # the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 0.03
    mass = 100 * unit.amu
    temperature = 300
    gpuid = f"{gpuid}"
    traj = basepath/f"{nchains}chains_N{N}_E0{E0}/run{run_number}"
    try:
        Path(traj).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Simulation directory already exists")
        return ran_sim
    
    if confine == "PBCbox":
        PBCbox = (L, L, L)
    else:
        PBCbox = False

    reporter = HDF5Reporter(folder=traj, max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA", 
        integrator="variableLangevin",
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N*nchains,
        save_decimals=2,
        PBCbox=PBCbox,
        reporters=[reporter],
    )
    #set lattice size to be 5 times the radius of a confined chain so that the chains
    #stay far apart from each other and don't interaction
    if resume:
        last_conf = list_URIs(traj.parent)[-1]
        polymer = load_URI(last_conf)["pos"]
    else:
        #polymer = starting_conformations.grow_cubic(N*nchains, 2 * int(np.ceil(r)))
        polymer = initialize_territories(volume_fraction, N, nchains)
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    f_sticky = forces.selective_SSW(sim, 
                                       sticky_inds, 
                                       extraHardParticlesIdxs=[], #don't make any particles extra hard
                                       repulsionEnergy=3.0, #base repulsion energy for all particles (same as polynomial_repulsive)
                                       attractionEnergy=0.0, #base attraction energy for all particles
                                       selectiveAttractionEnergy=E0)
    sim.add_force(f_sticky)
    if confine == "single":
        sim.add_force(forces.spherical_confinement(sim, r=r, k=5.0))
    elif confine == "many":
        sim.add_force(spherical_well_array(sim, cell_size=5*r_chain, r=width+r_chain, width=width, depth=depth))
    
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(i*N, i*N + N, False) for i in range(0, nchains)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=None,
            nonbonded_force_kwargs={},
            except_bonds=True,
        )
    )
    tic = time.perf_counter()
    if time_stepping_fn:
        time_stepping_fn(sim)
    else:
        for _ in range(nblocks):  # Do 10 blocks
            sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
    ran_sim = True
    return ran_sim

def short_time_dynamics(sim, stop1=2000*2000, block1=50, stop2=10000*2000, block2=5000):
    """Step until t=stop1 time steps with block size `block1`, and then step
    until `stop2` time steps  with block size `block2`."""
    nblocks = int(stop1 // block1)
    for _ in range(nblocks):
        sim.do_block(block1)
    nblocks = int((stop2 - stop1) // block2)
    for _ in range(nblocks):
        sim.do_block(block2)

def log_time_stepping(sim, ntimepoints=100, mint=50, maxt=10000*2000):
    """ Save data at time points that are log-spaced between t=mint and t=maxt."""
    timepoints = np.rint(np.logspace(np.log10(mint), np.log10(maxt), ntimepoints))
    blocks = np.concatenate(([timepoints[0]], np.diff(timepoints))).astype(int)
    for block in blocks:
        if block >= 1:
            sim.do_block(block)

if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    tic = time.time()
    f = 3
    N = 50
    nchains = 100
    E0 = 0.5
    sticky_ids = sticky_inds(N, f, 1)
    run_sticky_sim(gpuid, 0, N, nchains, E0, sticky_ids, volume_fraction=0.2, nblocks=100)
    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s") 
