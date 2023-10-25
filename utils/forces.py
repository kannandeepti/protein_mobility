r"""
Forces relevant to simulation of patched particles.
===================================================
(1) Repulsive potential between large spheres.
(2) Harmonic bonds between sphere and patches
(3) Angle forces enforcing a certain geometry of patches relative to sphere centroid.

"""
import numpy as np
try:
    import openmm
except Exception:
    import simtk.openmm as openmm
from polychrom import forces

def patched_particle_repulsive_group(sim_object, molecule_inds, trunc=5.0, radiusMult=1.0,
                         name="patched_particle_repulsive"):
    """
    Repulsive potential between spheres where patches on spheres are excluded.
    This version specifies an interaction group and only computes forces between
    the larger spheres and omits interactions between patches.

    This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    See the gist below with an example of the potential.
    https://gist.github.com/mimakaev/0327bf6ffe7057ee0e0625092ec8e318

    Parameters
    ----------
    molecule_inds : array-like[int] (N,)
        array of particle indices representing spheres (not patches)
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

    molecule_inds = [int(i) for i in molecule_inds]
    force.addInteractionGroup(molecule_inds, molecule_inds)
    for i in range(sim_object.N):
        force.addParticle([])
    force.setCutoffDistance(nbCutOffDist)

    return force

def patched_particle_repulsive(sim_object, moleculeIDs, trunc=5.0, radiusMult=1.0,
                         name="polynomial_repulsive"):
    """
    Repulsive potential between spheres where patches on spheres are excluded.
    This version still computes interactions between all pairs of particles and
    excludes patches by assigning a per particle parameter "molecule" which
    signifies if a particle is a sphere or a patch.

    This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    See the gist below with an example of the potential.
    https://gist.github.com/mimakaev/0327bf6ffe7057ee0e0625092ec8e318

    Parameters
    ----------
    moleculeIDs : array-like (N,)
        array of 0s and 1s where 1s indicate larger spheres and 0's are patches
    trunc : float
        the energy value around r=0

    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    repul_energy = (
        "min(molecule1, molecule2) * Erep;"
        "Erep = rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
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
    force.addPerParticleParameter("molecule")

    for i in range(sim_object.N):
        force.addParticle([float(moleculeIDs[i])])

    force.setCutoffDistance(nbCutOffDist)

    return force

def patched_particle_forcekit(
            sim_object,
            nchains,
            f,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={"bondWiggleDistance": 0.005, "bondLength": 0.5},
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": 0.05},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={"trunc": 5.0, "radiusMult": 1.0},
            except_bonds=False
    ):
        """Add bonds to create a topology where a centroid sphere is bonded
        to f other particles. Creates bonds and angle forces to enforce a particular geometry.
        TODO: implement dihedral forces for f>=4. how to keep them planar? Are there torques?

        Assumes every fth particle is the centroid of a patched particle.

        Parameters
        ----------
        nchains: int
            Number of patched particles in simulation.
        f : int
            valency of patches.
        except_bonds : bool
            If True then do not calculate non-bonded forces between the
            particles connected by a bond. False by default.

        """

        force_list = []
        bonds = []
        triplets = []
        centroids = []

        if f >= 4:
            raise ValueError("Have not implemented valencies larger than 4 yet.")

        for i in range(0, (f+1)*nchains, f+1):
            centroids.append(i)
            #create bonds between central atom and each of the f sticky residues
            bonds += [(i, i + j) for j in range(1, f+1)]
            #enforce angles between planar sticky residues
            #TODO: implement dihedrals for f >= 4
            for j in range(1, f + 1):
                for k in range(j + 1, f + 1):
                    triplets.append((i+j, i, i+k))

        report_dict = {
            "chains": np.array(centroids, dtype=int),
            "bonds": np.array(bonds, dtype=int),
            "angles": np.array(triplets),
        }
        for reporter in sim_object.reporters:
            reporter.report("forcekit_patched_spheres", report_dict)

        if bond_force_func is not None:
            force_list.append(bond_force_func(sim_object, bonds, **bond_force_kwargs))

        if angle_force_func is not None and f > 1:
            "equilibrium angles are planar for f < 4"
            theta = angle_force_kwargs["theta_0"] if "theta_0" in angle_force_kwargs else 2*np.pi/f
            angle_force_kwargs["theta_0"] = theta
            force_list.append(angle_force_func(sim_object, triplets, **angle_force_kwargs))

        if nonbonded_force_func is not None:
            nb_force = nonbonded_force_func(sim_object, **nonbonded_force_kwargs)

            if except_bonds:
                exc = list(set([tuple(i) for i in np.sort(np.array(bonds), axis=1)]))
                if hasattr(nb_force, "addException"):
                    print("Exclude bonded particles from {}".format(nb_force.name))
                    for pair in exc:
                        nb_force.addException(int(pair[0]), int(pair[1]), 0, 0, 0, True)

                    num_exc = nb_force.getNumExceptions()

                # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                elif hasattr(nb_force, "addExclusion"):
                    print("Exclude bonded particles from {}".format(nb_force.name))
                    nb_force.createExclusionsFromBonds([(int(b[0]), int(b[1])) for b in bonds],
                                                       int(except_bonds))
                    # for pair in exc:
                    #     nb_force.addExclusion(int(pair[0]), int(pair[1]))
                    num_exc = nb_force.getNumExclusions()

                print("Number of exceptions:", num_exc)

            force_list.append(nb_force)

        return force_list