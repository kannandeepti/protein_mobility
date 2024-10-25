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
from itertools import combinations
from .geometry import angles_from_patches

def patch_attraction(
    sim_object,
    patch_inds,
    attractionEnergy=3.0,  # base attraction energy for **all** particles
    attractionRadius=0.5,
    name="patch_attraction",
):
    """
    Negative of polynomial_repulsive -- makes a smoothed version of
    a square well potential that is attractive until attraction radius,
    which is also cutoff of force.

    Parameters
    ----------
    f : int
        Valency of patched particle
    patch_inds: list of int
        the list of indices of the "sticky" particles. The sticky particles
        are attracted to each other with `attractionEnergy`
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the maximal range of the attractive part of the potential.

    """

    energy = (
        "- rshft12 * (rshft2 - 1.0) * ATTRe / emin12 - ATTRe;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = r / ATTRrad * rmin12;"
        ""
    )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)
    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTRrad", attractionRadius * sim_object.conlen)
    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    patch_inds = [int(i) for i in patch_inds]
    force.addInteractionGroup(patch_inds, patch_inds)

    for i in range(sim_object.N):
        force.addParticle([])

    return force

def patched_particle_repulsive(sim_object, molecule_inds, trunc=5.0, radiusMult=1.0,
                         name="patched_particle_repulsive"):
    """
    Repulsive potential between spheres where patches on spheres are excluded.
    This version specifies an interaction group and only computes forces between
    the larger spheres and omits interactions between patches.

    This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    See the gist below with an example of the potential.
    https://gist.github.com/mimakaev/0327bf6ffe7057ee0e06250ls
    92ec8e318

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


def dihedral_force(sim_object, quadruplets, k=1.5, theta_0=np.pi, 
                   name="dihedral"):
    """Adds dihedral forces between a set of 4 particles. 
    k specifies energy in kT at one radian
    If k is an array, it has to be of the length N.

    Parameters
    ----------

    k : float or list of length N
        Stiffness of the bond.
        If list, then determines the stiffness of the i-th triplet
        Potential is k * alpha^2 * 0.5 * kT

    theta_0 : float or list of length N
        Equilibrium dihedral angle.

    """

    k = forces._to_array_1d(k, len(quadruplets))
    theta_0 = forces._to_array_1d(theta_0, len(quadruplets))

    energy = "kT*angK * (theta - angT0) * (theta - angT0) * (0.5)"
    force = openmm.CustomTorsionForce(energy)
    force.name = name

    force.addGlobalParameter("kT", sim_object.kT)
    force.addPerTorsionParameter("angK")
    force.addPerTorsionParameter("angT0")

    for quadruplet_idx, (p1, p2, p3, p4) in enumerate(quadruplets):
        force.addTorsion(
            int(p1),
            int(p2),
            int(p3),
            int(p4),
            (float(k[quadruplet_idx]), float(theta_0[quadruplet_idx])),
        )

    return force

def patched_particle_forcekit(
            sim_object,
            N,
            f,
            sticky_subset=None,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={"bondWiggleDistance": 0.005, "bondLength": 0.5},
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": 0.05},
            dihedral_force_func=dihedral_force,
            dihedral_force_kwargs={"k" : 0.05},
            patch_attraction_force_func=patch_attraction,
            patch_attraction_force_kwargs={'attractionEnergy' : 3.0,  'attractionRadius' : 0.5},
            nonbonded_force_func=patched_particle_repulsive,
            nonbonded_force_kwargs={"trunc": 30.0, "radiusMult": 1.05},
            exclude_intramolecular=True,
            except_bonds=False
    ):
        """Add bonds to create a topology where a centroid sphere is bonded
        to f other particles. Creates bonds and angle forces to enforce a particular geometry.
        TODO: implement dihedral forces for f>=4. how to keep them planar? Are there torques?

        Assumes every fth particle is the centroid of a patched particle.

        Parameters
        ----------
        N: int
            Number of patched particles in simulation.
        f : int
            valency of patches.
        sticky_subset : list
            subset of patched particles that can engage in disulfide bonding. (0-indexed) 
        exclude_intramolecular : bool
            If True then do not calculate patch attraction forces between patches on the same
            particle. True by default.
        except_bonds : bool
            If True then do not calculate non-bonded forces between the
            particles connected by a bond. False by default.

        """

        force_list = []
        bonds = []
        triplets = []
        quadruplets = []
        thetas = []
        dihedrals = []
        centroids = []

        particle_inds = np.arange(0, (f + 1) * N, 1)
        #indices of larger spheres
        molecule_inds = np.arange(0, (f + 1) * N, f + 1)
        #indices of patches that are sticky
        if sticky_subset is not None:
            sticky_molecule_inds = molecule_inds[sticky_subset]
            patch_inds = [ind + i for ind in sticky_molecule_inds for i in range(1, f+1)]
        else:
            patch_inds = np.setdiff1d(particle_inds, molecule_inds)

        angles_per_particle, dihedrals_per_particle = angles_from_patches(f)
        for i in range(0, (f+1)*N, f+1):
            centroids.append(i)
            #create bonds between central atom and each of the f sticky residues
            bonds += [(i, i + j) for j in range(1, f+1)]
            #enforce angles between sticky residues
            for pair in combinations(range(1, f+1), 2):
                triplets.append((i + pair[0], i, i + pair[1]))
            thetas += angles_per_particle
            #enforce dihedrals for f >=4
            if f >= 4:
                for quadruplet in combinations(range(1, f+1), 4):
                    quadruplets.append(tuple(i + j for j in quadruplet))
                dihedrals += dihedrals_per_particle

        assert(len(triplets) == len(thetas))
        assert(len(quadruplets) == len(dihedrals))
        report_dict = {
            "chains": np.array(centroids, dtype=int),
            "bonds": np.array(bonds, dtype=int),
            "angles": np.array(triplets),
        }
        for reporter in sim_object.reporters:
            reporter.report("forcekit_patched_spheres", report_dict)

        if bond_force_func is not None and len(bonds) > 0:
            force_list.append(bond_force_func(sim_object, bonds, **bond_force_kwargs))

        if angle_force_func is not None and f > 1:
            "equilibrium angles are planar for f < 4"
            thetas = angle_force_kwargs["theta_0"] if "theta_0" in angle_force_kwargs else thetas
            angle_force_kwargs["theta_0"] = thetas
            force_list.append(angle_force_func(sim_object, triplets, **angle_force_kwargs))

        if dihedral_force_func is not None and f >= 4:
            dihedrals = dihedral_force_kwargs["theta_0"] if "theta_0" in dihedral_force_kwargs else dihedrals
            dihedral_force_kwargs["theta_0"] = dihedrals
            force_list.append(dihedral_force_func(sim_object, quadruplets, **dihedral_force_kwargs))
        
        if patch_attraction_force_func is not None and len(patch_inds) > 0:
            patch_force = patch_attraction_force_func(sim_object, patch_inds,
                                                          **patch_attraction_force_kwargs)
            if exclude_intramolecular:
                # add exclusions for patches on the same particle
                for i in range(0, len(patch_inds), f):
                    for pair in combinations(patch_inds[i : i + f], 2):
                        patch_force.addExclusion(pair[0], pair[1])
            force_list.append(patch_force)

        if nonbonded_force_func is not None:
            nb_force = nonbonded_force_func(sim_object, molecule_inds, **nonbonded_force_kwargs)

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
