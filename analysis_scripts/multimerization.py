r"""" 
Cluster size distriubtion
=========================

Script to detect "bonded" clusters of patched particles. Uses DBScan on a precomputed distance matrix.
Two particles are considered neighbors if their center of masses are within a distance cutoff.
Further, at least one patch on one neighbor must be within a radius of 0.5 from a patch on its neighbor.
"""

import sklearn
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse import csr_matrix, coo_matrix
from matplotlib import pyplot as plt
import polychrom
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file

def particles_from_mols(mol_ids, f):
    """ Return particle IDs (including patches) from list of molecule IDs"""
    particles = []
    for mol in mol_ids:
        particles += [mol + i for i in range(f + 1)]
    return particles

def extract_trajectory(simdir, wrap=False, start=0, end=-1, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir.
    
    Parameters
    ----------
    simdir : str or Path
        path to simulation directory containing .h5 files
    wrap : bool
        Whether or not to wrap trajectorys back into box.
    start : int
        which time block to start loading conformations from
    end : int
        which time block to stop loading conformations from
    every_other : int
        skip every_other time steps when loading conformations
        
    Returns
    -------
    X : array_like (num_t, N, 3)
        x, y, z positions of all monomers over time
    
    """
    X = []
    data = list_URIs(simdir)
    #check if PBCbox was used
    initArgs = load_hdf5_file(Path(simdir)/"initArgs_0.h5")
    PBCbox = np.array(initArgs['PBCbox'])
    if PBCbox.any():
        boxsize = PBCbox
    if start == 0:
        starting_pos = load_hdf5_file(Path(simdir)/"starting_conformation_0.h5")['pos']
        X.append(starting_pos)
    for conformation in data[start:end:every_other]:
        pos = load_URI(conformation)['pos']
        if PBCbox.any() and wrap:
            mults = np.floor(pos / boxsize[None, :])
            pos = pos - mults * boxsize[None, :]
            assert pos.min() >= 0
        X.append(pos)
    X = np.array(X)
    if PBCbox.any():
        return X, boxsize
    else:
        return X

def pdist_PBC(X, boxsize):
    """ Pairwise distances with periodic boundary conditions. 
    Same output as scipy.spatial.distance.pdist. """
    N = X.shape[0]
    dim = X.shape[1]
    dist_nd_sq = np.zeros(N * (N - 1) // 2)  # to match the result of pdist
    for d in range(dim):
        pos_1d = X[:, d][:, np.newaxis]  # shape (N, 1)
        dist_1d = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
        dist_1d -= boxsize[d] * np.rint(dist_1d / boxsize[d])
        dist_nd_sq += dist_1d ** 2  # d^2 = dx^2 + dy^2 + dz^2
    dist_nd = np.sqrt(dist_nd_sq)
    return dist_nd

def cdist_PBC(X, Y, boxsize):
    """ Pairwise distances with periodic boundary conditions. 
    Same output as scipy.spatial.distance.cdist. """
    cdist_nd_sq = np.zeros((X.shape[0], Y.shape[0]))
    dim = len(boxsize)
    for d in range(dim):
        xpos_1d = X[:, d][:, np.newaxis]  # shape (N, 1)
        ypos_1d = Y[:, d][:, np.newaxis]  # shape (M, 1)
        dist_1d = cdist(xpos_1d, ypos_1d)  # shape (N, M)
        dist_1d -= boxsize[d] * np.rint(dist_1d / boxsize[d])
        cdist_nd_sq += dist_1d ** 2  # d^2 = dx^2 + dy^2 + dz^2
    dist_nd = np.sqrt(cdist_nd_sq)
    return dist_nd

def cluster_size_distribution_pruned(datadir, f, E0, N=1000, vol_fraction=0.3, r=0.35, rep_r=1.1,
                              molecule_cutoff=1.3, patch_cutoff=0.55, wrap=False):
    """ Compute cluster size distribution from 500 different snapshots in the steady state portion
    of the trajectory. First enforces that proteins are within `molecule_cutoff` distance of each other.
    Then checks that at least one pair of patches on adjacent molecules are within `patch_cutoff`
    distance of one another.
    
    Parameters
    ----------
    datadir : Path or str
        path to simulation results
    f : int
        valency of patched particles
    E0 : float
        patch-patch attraction energy
    N : int
        number of molecules. Defaults to 1000.
    vol_fraction : float
        Volume fraction of molecules in box. Defaults to 0.3.
    r : float
        patch-patch attraction range
    rep_r : float
        molecule-molecule repulsion range
    molecule_cutoff : float
        centroid-to-centroid distance that defines neighboring particles.
    patch_cutoff : float
        patch-to-patch distance that defines "bonded" patches.
    wrap : bool
        Whether to wrap trajectory before computing distances (should not change answer). Defaults to False.

    Returns
    -------
    cluster_sizes : list
        List of all cluster sizes from all snapshots. Can be used to construct or plot a histogram.
    cluster_histogram : (1000,) array
        m^th element contains number of monomers in a (m+1)-mer. 
        Ex: if there were 4 dimers, cluster_histogram[1] = 4 * 2 = 8.

    
    """
    Y, boxsize = extract_trajectory(Path(datadir)/f"N{N}_f{f}_E0{E0}_v{vol_fraction}_r{r}_rep{rep_r}", 
                           wrap=wrap, start=10000, every_other=10)
    #indices of larger spheres
    molecule_inds = np.arange(0, (f+1)*N, f+1)
    cluster_sizes = []
    cluster_histogram = np.zeros(1000)
    for i in range(Y.shape[0]):
        X = Y[i, molecule_inds, :]
        distances = pdist_PBC(X, boxsize)
        dist_graph = squareform(distances)
        dist_graph[dist_graph > molecule_cutoff] = 0.0 #dist_graph is dense and symmetric
        dist_graph_lower = dist_graph.copy()
        #set the lower triangle to zero since its redundant
        lower_triangle_indices = np.tril_indices(dist_graph_lower.shape[0], k=-1)
        dist_graph_lower[lower_triangle_indices] = 0.0
        #sparse version of dist_graph with no redundant edges
        dist_graph_sparse = coo_matrix(dist_graph_lower)
        #prune edges
        for i, (r, c, d) in enumerate(zip(dist_graph_sparse.row, dist_graph_sparse.col, dist_graph_sparse.data)):
            node1_patches = np.array([Y[-1, r*(f+1) + k, :] for k in range(1, f+1)])
            node2_patches = np.array([Y[-1, c*(f+1) + k, :] for k in range(1, f+1)])
            inter_patch_distances = cdist_PBC(node1_patches, node2_patches, boxsize)
            if np.all(inter_patch_distances > patch_cutoff):
                dist_graph[r, c] = 0.0
                dist_graph[c, r] = 0.0
        if np.sum(dist_graph > 0) == 0:
            #all the edges were pruned. 
            cluster_sizes += dist_graph.shape[0] * [1]
            cluster_histogram[0] += dist_graph.shape[0]
            continue
        dist_graph_sparse = csr_matrix(dist_graph)
        dist_graph_sparse = sklearn.neighbors.sort_graph_by_row_values(dist_graph_sparse,
                                warn_when_not_sorted=False)
        clustering = DBSCAN(eps=molecule_cutoff, min_samples=2, metric="precomputed").fit(dist_graph_sparse)
        cluster_labels, counts = np.unique(clustering.labels_, return_counts=True)
        #first element of counts represents number of monomers
        cluster_sizes += counts[0] * [1] + list(counts[1:])
        cluster_histogram[0] += counts[0]
        #multimers contains the unique cluster sizes found in this snapshot (2, 3, etc.)
        #number_observed is the number of each multimer observed
        multimers, number_observed = np.unique(counts[1:], return_counts=True)
        for m in range(len(multimers)):
            cluster_histogram[multimers[m] - 1] += number_observed[m] * multimers[m]
    return cluster_sizes, cluster_histogram 
