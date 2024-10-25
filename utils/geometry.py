r"""
Geometry of patches relative to sphere centroid
===============================================
Determine positions of f patches relative to sphere
centroid such that the patches are roughly uniformly
distributed around the sphere. Also compute angles
and dihedrals required for each valency to be used in
angle forces.

"""

import numpy as np
from itertools import combinations

def patched_particle_geom(f, R=1):
    """ Distribute f patches on a sphere with equal angles."""

    if f <= 3:
        #put 1, 2, or 3 patches in a plane arranged in the xy plane
        # first position is center particle
        positions = [[0., 0., 0.]]
        theta = np.pi / 2
        for i in range(f):
            # for valency less than 5, just distribute points on a circle in the x-y plane
            phi = 2 * np.pi * i / f
            x = R * np.sin(theta) * np.cos(phi)
            y = R * np.sin(theta) * np.sin(phi)
            z = R * np.cos(theta)
            positions.append([x, y, z])
        return np.array(positions)

    if f == 4:
        #place 4 patches at vertices of a tetrahedron centered at origin with lower face
        #parallel to xy plane
        positions = np.array([[0., 0., 0.],
                              [np.sqrt(8/9), 0.0, -1/3],
                              [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                              [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
                              [0.0, 0.0, 1.0]])
        return R * np.array(positions)

    if f == 5 or f == 6:
        # place 6 patches at vertices of an octahedron centered at the origin
        positions = np.array([[0., 0., 0.],
                              [1., 0., 0.],
                              [-1., 0., 0.],
                              [0., 1., 0.],
                              [0., -1., 0.],
                              [0., 0., 1.],
                              [0., 0., -1.]])
        return R * positions[0:(f+1), :]

    if f >= 7:
        raise ValueError("Have not implemented f >= 7 yet")

def dihedral(p):
    """ Calculate the dihedral angle given a set of 4 points. """
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x )

def angles_from_patches(f):
    """ Determine the angles between pairs of patches and the dihedral angles
    among triplets of patches for a particular valency f."""
    positions = patched_particle_geom(f, R=1)
    angles = []
    dihedrals = []
    for pair in combinations(range(1, f+1), 2):
        #for each pair of patches, compute angle between them
        angles.append(np.arccos(np.clip(np.dot(positions[pair[0]], positions[pair[1]]), -1.0, 1.0)))
    
    for quadruplet in combinations(range(1, f+1), 4):
        dihedrals.append(dihedral(positions[list(quadruplet)]))
        
    return angles, dihedrals

