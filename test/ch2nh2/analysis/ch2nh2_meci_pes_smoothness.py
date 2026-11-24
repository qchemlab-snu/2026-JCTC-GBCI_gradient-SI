"""Are the CH2NH2+ S0/S1 surfaces smooth around the MECI?

Second half of the referee question on protonated formaldimine: besides the
character of the crossing states (``ch2nh2_meci_ci_weights.py``), the referee
asks about the smoothness of the potential energy surfaces around the MECI.
The concern specific to GBCI is that every configuration group carries its own
relaxed bath, so a surface could in principle kink where a bath solution moves
-- something CASCI (one set of RHF orbitals) and SA-CASSCF (one state-averaged
set) cannot do.

The probe is a straight line in Cartesian space through both seams::

    R(s) = R_GBCI + s * (R_SA-CASSCF,aligned - R_GBCI)

sampled from ``S_MIN`` to ``S_MAX``, so s = 0 is the GBCI MECI and s = 1 the
SA-CASSCF MECI, with the scan running past both ends.  The SA-CASSCF structure
is first superposed on the GBCI one (Kabsch), otherwise the line would contain
a rigid rotation.

At every point each method is run as its own MECI optimization was, and the
quantities reported are

* ``E0``, ``E1`` and the gap -- the gap has a physical cusp at a conical
  intersection, in every method,
* the mean energy ``(E0 + E1) / 2``, which is *not* cusped at a conical
  intersection and is therefore the quantity whose smoothness is diagnostic,
* its discrete second difference along the line, which a bath discontinuity
  would show as a spike against the smooth background of CASCI/SA-CASSCF,
* the dominant configuration, its weight and the Loewdin CH2 charge of each
  state, so a diabatic switch of character is visible as such rather than as a
  kink.

Single-point energies only: no gradients are computed, so each scanner's
gradient step is stubbed out.

Usage::

    python3 ch2nh2_meci_pes_smoothness.py                 # all three methods
    python3 ch2nh2_meci_pes_smoothness.py GBCI CASCI      # a subset
"""

import json
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.interpolate import CubicSpline
from pyscf import gto, mcscf
from pyscf.fci import direct_spin1

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ci_weights import (
    configuration_ci_vectors,
    configuration_weights,
    format_occ,
    group_configuration_labels,
    group_weights,
)
from utils.geometry_descriptors import (
    kabsch_align,
    pyramidalization_angle,
    read_xyz,
    torsion_angle,
)
from utils.meci_optimizer import (
    CASCIActiveRootTrackedPairScanner,
    CASSCFActiveRootTrackedPairScanner,
    GBCIActiveRootTrackedPairScanner,
    HARTREE2EV,
)

RUN_NAME = "ch2nh2_ccpvtz_meci_pes_smoothness"

BASIS = "cc-pVTZ"
CHARGE = 1
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 0

# (2e,2o), the active space of the MECI runs being compared.
NCAS = 2
NELECAS = (1, 1)
ACT_LIST = [8, 9]
ACT_BASE = 1
GROUPA = [[0], [1]]

# CASCI and GBCI run three roots; SA-CASSCF averages over the two crossing
# states with equal weights, exactly as ch2nh2_casscf_S0S1_meci_2o2e_ccpvtz.py.
METHOD_NROOTS = {"GBCI": 3, "CASCI": 3, "SA-CASSCF": 2}
TARGET_ROOTS = (0, 1)
CONV_TOL = 1e-10
MAX_CYCLE = 200

CONF_TOL = 5.0e-3

MECI_XYZ = {
    "CASCI": ("../casci/outputs/ch2nh2_casci_S0S1_meci_2o2e_ccpvtz_sigma0.25/"
              "ch2nh2_casci_S0S1_meci_2o2e_ccpvtz_sigma0.25_optimized.xyz"),
    "GBCI": ("../gbci/outputs/ch2nh2_gbci_S0S1_meci_2o2e_groupocc_ccpvtz/"
             "ch2nh2_gbci_S0S1_meci_2o2e_groupocc_ccpvtz_optimized.xyz"),
    "SA-CASSCF": ("../casscf/outputs/ch2nh2_casscf_S0S1_meci_2o2e_ccpvtz/"
                  "ch2nh2_casscf_S0S1_meci_2o2e_ccpvtz_optimized.xyz"),
}

# Paths through the MECI structures.  The coordinate is the signed arc length in
# Angstrom with zero at ``origin`` (the GBCI MECI in both), the legs are sampled
# at a near-constant Cartesian step, and ``extend_angstrom`` continues the path
# straight past both end points.  A piecewise-linear path turns a corner at an
# interior waypoint -- here the two legs meet at 62 degrees -- so the second
# derivative of ANY quantity has a spike there by construction.  That spike is a
# property of the path, identical for every method, which is why the methods are
# compared with each other at that point rather than against zero.
PATHS = {
    "gbci_casscf": {
        "waypoints": [("GBCI", MECI_XYZ["GBCI"]),
                      ("SA-CASSCF", MECI_XYZ["SA-CASSCF"])],
        "origin": "GBCI",
        "step_angstrom": 0.0114,
        "extend_angstrom": 0.057,
    },
    "casci_gbci_casscf": {
        "waypoints": [("CASCI", MECI_XYZ["CASCI"]),
                      ("GBCI", MECI_XYZ["GBCI"]),
                      ("SA-CASSCF", MECI_XYZ["SA-CASSCF"])],
        "origin": "GBCI",
        "mode": "linear",
        "step_angstrom": 0.025,
        "extend_angstrom": 0.10,
    },
    # The same three structures on a curve of constant curvature instead of two
    # straight legs: no corner and no curvature maximum at the GBCI MECI, so
    # nothing in the plots can be mistaken for one.  This is the path to show.
    "casci_gbci_casscf_smooth": {
        "waypoints": [("CASCI", MECI_XYZ["CASCI"]),
                      ("GBCI", MECI_XYZ["GBCI"]),
                      ("SA-CASSCF", MECI_XYZ["SA-CASSCF"])],
        "origin": "GBCI",
        "mode": "arc",
        "step_angstrom": 0.025,
        "extend_angstrom": 0.10,
    },
}
DEFAULT_PATH = "casci_gbci_casscf_smooth"

# Atom order of both files: N, C, H(N), H(N), H(C), H(C).
NH2_ATOMS = [0, 2, 3]
CH2_ATOMS = [1, 4, 5]

METHODS = ("GBCI", "SA-CASSCF", "CASCI")


def _zero_gradients(natm, nroots=2):
    return np.zeros((nroots, natm, 3))


class _EnergyOnlyGBCIScanner(GBCIActiveRootTrackedPairScanner):
    """GBCI pair scanner without the gradient step (energies only)."""

    def compute_gbci_gradients(self, gbci, mo_cas, roots, intermediates):
        return _zero_gradients(gbci.mol.natm, len(roots))


class _EnergyOnlyCASSCFScanner(CASSCFActiveRootTrackedPairScanner):
    """SA-CASSCF pair scanner without the gradient step (energies only)."""

    def compute_gradients(self, mc, roots):
        return _zero_gradients(mc.mol.natm, len(roots))


def _casci_without_gradients(mf, ncas, nelecas, nroots):
    """CASCI object whose nuc_grad_method returns zeros.

    The CASCI scanner computes its gradient inline rather than through an
    overridable method, so the stub goes on the CASCI object itself.
    """
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.nroots = nroots

    class _NoGradient:
        def kernel(self, *args, **kwargs):
            return np.zeros((mf.mol.natm, 3))

    mc.nuc_grad_method = lambda *args, **kwargs: _NoGradient()
    return mc


def build_scanner(method):
    nroots = METHOD_NROOTS[method]
    common = dict(
        ncas=NCAS,
        nelecas=NELECAS,
        act_list=ACT_LIST,
        act_base=ACT_BASE,
        nroots=nroots,
        target_roots=TARGET_ROOTS,
        conv_tol=CONV_TOL,
        max_cycle=MAX_CYCLE,
        root_tracking=False,
        active_tracking=False,
        verbose=False,
    )
    if method == "GBCI":
        return _EnergyOnlyGBCIScanner(groupA=GROUPA, **common)
    if method == "SA-CASSCF":
        return _EnergyOnlyCASSCFScanner(
            state_weights=[1.0 / nroots] * nroots, **common
        )
    if method == "CASCI":
        return CASCIActiveRootTrackedPairScanner(
            mc_factory=_casci_without_gradients, **common
        )
    raise ValueError(f"unknown method {method!r}")


def lowdin_matrix(mol):
    s = mol.intor_symmetric("int1e_ovlp")
    return scipy.linalg.sqrtm(s).real


def lowdin_atom_populations(mol, dm_ao, s12):
    """Loewdin population of ``dm_ao`` on each atom, in electrons."""
    pop = np.diag(reduce(np.dot, (s12, dm_ao, s12))).real
    out = np.zeros(mol.natm)
    slices = mol.aoslice_by_atom()
    for atom in range(mol.natm):
        p0, p1 = slices[atom][2], slices[atom][3]
        out[atom] = pop[p0:p1].sum()
    return out


def lowdin_atom_charges(mol, dm_ao, s12):
    pop = lowdin_atom_populations(mol, dm_ao, s12)
    return np.asarray([mol.atom_charge(a) for a in range(mol.natm)]) - pop


def state_density_ao(method, mc, ci_root, ncore):
    if method == "GBCI":
        return mc.make_rdm1(ci_root)
    casdm1 = direct_spin1.make_rdm1(ci_root, NCAS, mc.nelecas)
    mo = mc.mo_coeff
    mocore = mo[:, :ncore]
    mocas = mo[:, ncore:ncore + NCAS]
    return 2.0 * mocore.dot(mocore.T) + mocas.dot(casdm1).dot(mocas.T)


def _load_waypoint_structures(spec):
    """Read the MECI structures of ``spec`` and superpose them on the first."""
    labels = [label for label, _ in spec["waypoints"]]
    symbols = None
    coords_list = []
    rmsds = []
    for label, rel_path in spec["waypoints"]:
        syms, coords = read_xyz((SCRIPT_DIR / rel_path).resolve())
        if symbols is None:
            symbols = syms
            coords_list.append(coords)
            rmsds.append(0.0)
            continue
        if syms != symbols:
            raise ValueError(f"{label}: atom order differs from {labels[0]}")
        aligned, rmsd = kabsch_align(coords_list[0], coords)
        coords_list.append(aligned)
        rmsds.append(rmsd)
    return labels, symbols, coords_list, rmsds


def build_path_spline(spec):
    """Smooth (C2) curve through the MECI structures of ``spec``.

    A piecewise-linear path turns a corner at every interior waypoint, and the
    corner alone puts a spike in the second derivative of anything measured
    along it -- which is exactly what one must not confuse with a kink in the
    surface.  This builder removes that artefact: a natural cubic spline is
    fitted through the structures in Cartesian space (one spline per coordinate,
    chord length as the parameter), so the curve has continuous curvature
    everywhere, the interior waypoints included.  Its own cubic extrapolation
    continues it past the end points, so the end waypoints are not corners
    either.

    The points are then placed at a near-constant arc length, each segment
    between two waypoints getting a whole number of steps so that every
    waypoint lands exactly on a sampled point.  ``path_curvature`` of each point
    is |d2R/ds2| of the curve itself, in 1/Angstrom: it says how much of any
    residual structure in d2E/ds2 the path itself can account for.
    """
    labels, symbols, coords_list, rmsds = _load_waypoint_structures(spec)
    if len(coords_list) < 3:
        raise ValueError("a spline path needs at least three structures")

    natm = coords_list[0].shape[0]
    flat = np.asarray([c.reshape(-1) for c in coords_list])

    # Chord length is the usual well-behaved parameter for an interpolating
    # spline through scattered points.
    chord = np.zeros(len(flat))
    for i in range(1, len(flat)):
        chord[i] = chord[i - 1] + float(np.linalg.norm(flat[i] - flat[i - 1]))

    spline = CubicSpline(chord, flat, axis=0, bc_type="natural", extrapolate=True)
    d1 = spline.derivative(1)
    d2 = spline.derivative(2)

    step = spec["step_angstrom"]
    extend = spec["extend_angstrom"]

    # Arc length s(t) of the curve, on a dense grid that also covers the
    # extensions, by cumulative trapezoid of |dR/dt|.
    t_pad = extend * 2.0 + step
    t_dense = np.linspace(chord[0] - t_pad, chord[-1] + t_pad, 20001)
    speed = np.linalg.norm(d1(t_dense), axis=1)
    s_dense = np.concatenate(
        [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t_dense))]
    )
    s_of_t = lambda t: np.interp(t, t_dense, s_dense)
    t_of_s = lambda s: np.interp(s, s_dense, t_dense)

    s_way = s_of_t(chord)
    origin_index = labels.index(spec["origin"])
    s_origin = float(s_way[origin_index])

    # Sample each segment with a whole number of steps, then extend both ends.
    s_targets = []
    for i in range(len(s_way) - 1):
        seg = float(s_way[i + 1] - s_way[i])
        n_int = max(1, int(round(seg / step)))
        s_targets.extend(s_way[i] + seg * np.arange(n_int) / n_int)
    s_targets.append(float(s_way[-1]))
    h_front = (s_way[1] - s_way[0]) / max(1, int(round((s_way[1] - s_way[0]) / step)))
    h_back = (s_way[-1] - s_way[-2]) / max(
        1, int(round((s_way[-1] - s_way[-2]) / step))
    )
    front = [s_way[0] - h_front * k
             for k in range(max(0, int(round(extend / h_front))), 0, -1)]
    back = [s_way[-1] + h_back * k
            for k in range(1, max(0, int(round(extend / h_back))) + 1)]
    s_targets = np.asarray(front + s_targets + back, dtype=float)

    t_targets = t_of_s(s_targets)
    coords_all = spline(t_targets)
    r1 = d1(t_targets)
    r2 = d2(t_targets)
    speed_t = np.linalg.norm(r1, axis=1)
    tangent = r1 / speed_t[:, None]
    # d2R/ds2 = (R'' - (R''.T) T) / |R'|^2
    normal = (r2 - (np.sum(r2 * tangent, axis=1)[:, None] * tangent))
    curvature = np.linalg.norm(normal, axis=1) / speed_t**2

    path = [
        (float(s - s_origin), coords_all[k].reshape(natm, 3))
        for k, s in enumerate(s_targets)
    ]
    waypoints = [
        {"label": label, "s": float(s_way[i] - s_origin),
         "rmsd_to_first_angstrom": float(rmsds[i]),
         "file": spec["waypoints"][i][1]}
        for i, label in enumerate(labels)
    ]
    leg_info = [
        {"from": labels[i], "to": labels[i + 1],
         "length_angstrom": float(s_way[i + 1] - s_way[i]),
         "n_intervals": max(1, int(round(float(s_way[i + 1] - s_way[i]) / step))),
         "step_angstrom": float(s_way[i + 1] - s_way[i])
         / max(1, int(round(float(s_way[i + 1] - s_way[i]) / step)))}
        for i in range(len(s_way) - 1)
    ]
    return symbols, path, waypoints, leg_info, [float(x) for x in curvature]


def build_path_arc(spec):
    """Circular arc through exactly three MECI structures.

    Three points span a plane and lie on one circle in it, and a circle has
    *constant* curvature.  That is what makes this the path to show: a
    piecewise-linear path has a corner at the middle structure, and even a
    spline through the three has a curvature maximum there, so in both cases the
    second derivative of anything measured along the path has a feature at the
    GBCI MECI that belongs to the path and not to the surfaces.  On the arc no
    point is geometrically special, so whatever structure is left in d2<E>/ds2
    is the surfaces' own.

    The price is that the arc bulges away from the straight line between two
    structures (here by up to ~0.11 Angstrom, visible as r(CN) rising to 1.51
    Angstrom between the CASCI and GBCI structures).  The path is a probe, not a
    reaction path, and it still passes exactly through all three MECI
    geometries.
    """
    labels, symbols, coords_list, rmsds = _load_waypoint_structures(spec)
    if len(coords_list) != 3:
        raise ValueError("an arc path needs exactly three structures")

    natm = coords_list[0].shape[0]
    flat = [c.reshape(-1) for c in coords_list]
    origin_index = labels.index(spec["origin"])

    # Orthonormal basis of the plane through the three points.
    u = flat[1] - flat[0]
    v = flat[2] - flat[0]
    e1 = u / np.linalg.norm(u)
    w = v - (v @ e1) * e1
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1.0e-8 * float(np.linalg.norm(v)):
        raise ValueError("the three structures are collinear; use mode 'linear'")
    e2 = w / w_norm

    p2 = [np.array([0.0, 0.0]),
          np.array([u @ e1, u @ e2]),
          np.array([v @ e1, v @ e2])]
    (ax_, ay_), (bx_, by_), (cx_, cy_) = p2
    det = 2.0 * (ax_ * (by_ - cy_) + bx_ * (cy_ - ay_) + cx_ * (ay_ - by_))
    center = np.array([
        ((ax_**2 + ay_**2) * (by_ - cy_) + (bx_**2 + by_**2) * (cy_ - ay_)
         + (cx_**2 + cy_**2) * (ay_ - by_)) / det,
        ((ax_**2 + ay_**2) * (cx_ - bx_) + (bx_**2 + by_**2) * (ax_ - cx_)
         + (cx_**2 + cy_**2) * (bx_ - ax_)) / det,
    ])
    radius = float(np.linalg.norm(p2[0] - center))

    angles = [float(np.arctan2(q[1] - center[1], q[0] - center[0])) for q in p2]
    ref = angles[origin_index]
    unwrapped = []
    for a in angles:
        while a - ref > np.pi:
            a -= 2.0 * np.pi
        while a - ref < -np.pi:
            a += 2.0 * np.pi
        unwrapped.append(a)
    direction = 1.0 if unwrapped[2] > unwrapped[0] else -1.0
    s_way = [direction * (a - ref) * radius for a in unwrapped]
    if not (s_way[0] < s_way[1] < s_way[2]):
        raise ValueError("the three structures are not in order around the circle")

    def coords_at(s):
        angle = ref + direction * s / radius
        q = center + radius * np.array([np.cos(angle), np.sin(angle)])
        return (flat[0] + q[0] * e1 + q[1] * e2).reshape(natm, 3)

    step = spec["step_angstrom"]
    extend = spec["extend_angstrom"]

    s_targets = []
    n_ints = []
    for i in range(2):
        seg = s_way[i + 1] - s_way[i]
        n_int = max(1, int(round(seg / step)))
        n_ints.append(n_int)
        s_targets.extend(s_way[i] + seg * np.arange(n_int) / n_int)
    s_targets.append(s_way[2])
    h_front = (s_way[1] - s_way[0]) / n_ints[0]
    h_back = (s_way[2] - s_way[1]) / n_ints[1]
    front = [s_way[0] - h_front * k
             for k in range(max(0, int(round(extend / h_front))), 0, -1)]
    back = [s_way[2] + h_back * k
            for k in range(1, max(0, int(round(extend / h_back))) + 1)]
    s_targets = front + list(s_targets) + back

    path = [(float(s), coords_at(s)) for s in s_targets]
    waypoints = [
        {"label": label, "s": float(s_way[i]),
         "rmsd_to_first_angstrom": float(rmsds[i]),
         "file": spec["waypoints"][i][1]}
        for i, label in enumerate(labels)
    ]
    leg_info = [
        {"from": labels[i], "to": labels[i + 1],
         "length_angstrom": float(s_way[i + 1] - s_way[i]),
         "n_intervals": n_ints[i],
         "step_angstrom": float(s_way[i + 1] - s_way[i]) / n_ints[i]}
        for i in range(2)
    ]
    curvature = [1.0 / radius] * len(path)
    return symbols, path, waypoints, leg_info, curvature


def build_path(spec):
    """Path through the MECI structures of ``spec``; see ``spec['mode']``."""
    mode = spec.get("mode", "linear")
    if mode == "arc":
        return build_path_arc(spec)
    if mode == "spline":
        return build_path_spline(spec)
    return build_path_linear(spec)


def build_path_linear(spec):
    """Piecewise-linear path through the MECI structures of ``spec``.

    Every structure is superposed on the first one (Kabsch) so the path carries
    no rigid rotation, the legs are sampled at a near-constant Cartesian step,
    and the path is extended past both end points.  The path coordinate is the
    signed arc length in Angstrom with zero at the waypoint named
    ``spec["origin"]``, so the same coordinate means the same thing in every
    path this script can run.

    Returns ``(symbols, path, waypoints)`` where ``path`` is a list of
    ``(s, coords)`` and ``waypoints`` describes each structure.
    """
    labels, symbols, coords_list, rmsds = _load_waypoint_structures(spec)

    step = spec["step_angstrom"]
    extend = spec["extend_angstrom"]

    legs = []
    for i in range(len(coords_list) - 1):
        direction = coords_list[i + 1] - coords_list[i]
        length = float(np.linalg.norm(direction))
        n_int = max(1, int(round(length / step)))
        legs.append({"start": i, "direction": direction, "length": length,
                     "n_int": n_int, "h": length / n_int})

    # Arc length of every waypoint, then the sampling points.
    arc = [0.0]
    for leg in legs:
        arc.append(arc[-1] + leg["length"])
    origin_index = labels.index(spec["origin"])
    arc = [a - arc[origin_index] for a in arc]

    points = []  # (s, coords)
    n_ext_front = max(0, int(round(extend / legs[0]["h"])))
    for k in range(n_ext_front, 0, -1):
        frac = -k / legs[0]["n_int"]
        points.append((arc[0] + frac * legs[0]["length"],
                       coords_list[0] + frac * legs[0]["direction"]))
    for i, leg in enumerate(legs):
        # The last point of a leg is the first point of the next one, so it is
        # emitted once, by the leg that ends there (or by the tail extension).
        for k in range(leg["n_int"]):
            frac = k / leg["n_int"]
            points.append((arc[i] + frac * leg["length"],
                           coords_list[i] + frac * leg["direction"]))
    points.append((arc[-1], coords_list[-1]))
    last = legs[-1]
    n_ext_back = max(0, int(round(extend / last["h"])))
    for k in range(1, n_ext_back + 1):
        frac = k / last["n_int"]
        points.append((arc[-1] + frac * last["length"],
                       coords_list[-1] + frac * last["direction"]))

    waypoints = [
        {"label": label, "s": float(arc[i]),
         "rmsd_to_first_angstrom": float(rmsds[i]),
         "file": spec["waypoints"][i][1]}
        for i, label in enumerate(labels)
    ]
    leg_info = [
        {"from": labels[leg["start"]], "to": labels[leg["start"] + 1],
         "length_angstrom": leg["length"], "n_intervals": leg["n_int"],
         "step_angstrom": leg["h"]}
        for leg in legs
    ]
    path = [(float(s), coords) for s, coords in points]
    # A straight leg has zero curvature; the corners are between the samples.
    return symbols, path, waypoints, leg_info, [0.0] * len(path)


def make_mol(symbols, coords):
    atom = [(sym, tuple(xyz)) for sym, xyz in zip(symbols, coords)]
    return gto.M(atom=atom, basis=BASIS, charge=CHARGE, spin=SPIN, unit=UNIT,
                 verbose=VERBOSE)


def point_record(method, scanner, mol, s):
    result = scanner(mol)
    mc = result.mc
    ncore = int(mc.ncore)
    ci_list = scanner._as_ci_list(mc)

    e_states = getattr(mc, "e_states", None)
    if e_states is not None and np.size(e_states) >= len(ci_list):
        e_tot = np.asarray(e_states, dtype=float).reshape(-1)
    else:
        e_tot = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))

    order = np.argsort(e_tot[:len(ci_list)])  # label states adiabatically
    s12 = lowdin_matrix(mol)

    states = []
    group_info = None
    mo_list = None
    if method == "GBCI":
        group_info = np.asarray(
            scanner.last_gbci_intermediates["conf_info_list"], dtype=int
        )
        # One set of relaxed orbitals per bath group; its first ncore columns
        # are that group's core, which is what the active density is taken
        # against below.
        mo_list = np.asarray(scanner.last_gbci_intermediates["mo_list"])

    for rank in range(2):
        root = int(order[rank])
        ci_root = ci_list[root]
        confs = configuration_weights(ci_root, NCAS, NELECAS, tol=CONF_TOL)
        dm = state_density_ao(method, mc, ci_root, ncore)
        q = lowdin_atom_charges(mol, dm, s12)
        entry = {
            "root": root,
            "energy": float(e_tot[root]),
            "q_CH2": float(q[CH2_ATOMS].sum()),
            "q_NH2": float(q[NH2_ATOMS].sum()),
            # Per-atom charges as well: the nitrogen alone carries the lone
            # pair / pi hole whose occupation distinguishes the two diabats,
            # and the fragment sum hides it behind the two amine hydrogens.
            "q_N": float(q[0]),
            "q_atoms": [float(x) for x in q],
            "top_conf": format_occ(confs[0]["occ"]) if confs else "-",
            "top_weight": float(confs[0]["weight"]) if confs else 0.0,
            "configurations": confs,
        }
        if group_info is not None:
            entry["group_weights"] = [
                float(x) for x in group_weights(ci_root, group_info)
            ]
        states.append(entry)

    e0, e1 = states[0]["energy"], states[1]["energy"]
    record = {
        "s": float(s),
        "method": method,
        "E0": e0,
        "E1": e1,
        "E_mean": 0.5 * (e0 + e1),
        "gap_eV": float((e1 - e0) * HARTREE2EV),
        "states": states,
    }
    if group_info is not None:
        record["group_labels"] = group_configuration_labels(
            group_info, NCAS, NELECAS
        )
        # The charge of each configuration itself, not of an eigenstate: the
        # density of the pure configuration in its own relaxed bath.  This is
        # the only way to see the configurations the eigenstates never lead
        # with -- "0 2" here -- and it is free of the adiabatic relabelling.
        configuration_states = []
        for label, civec in configuration_ci_vectors(NCAS, NELECAS):
            igroup = int(np.argmax(group_weights(civec, group_info)))
            dm = mc.make_rdm1(civec)
            q = lowdin_atom_charges(mol, dm, s12)
            # Active-space part of the density: the total minus this group's
            # own doubly occupied core.  A pure single-group CI vector is a
            # single determinant in that group's orbitals, so the subtraction
            # is exact and the remainder carries exactly nelecas electrons.
            ccore = mo_list[igroup][:, :ncore]
            dm_act = dm - 2.0 * ccore.dot(ccore.T)
            pop_act = lowdin_atom_populations(mol, dm_act, s12)
            configuration_states.append({
                "occ": label,
                "group": igroup,
                "q_CH2": float(q[CH2_ATOMS].sum()),
                "q_NH2": float(q[NH2_ATOMS].sum()),
                "q_N": float(q[0]),
                "q_atoms": [float(x) for x in q],
                "pop_act_N": float(pop_act[0]),
                "pop_act_NH2": float(pop_act[NH2_ATOMS].sum()),
                "pop_act_CH2": float(pop_act[CH2_ATOMS].sum()),
                "pop_act_atoms": [float(x) for x in pop_act],
                "pop_act_total": float(pop_act.sum()),
            })
        record["configuration_states"] = configuration_states
    return record


def report_diabatic(payload):
    """Re-read a finished scan and sort the two states by configuration.

    The adiabatic labels swap at the seam, so the surfaces that should be smooth
    *through* it are the configuration-resolved ones: the closed-shell ``2 0``
    diabat and the open-shell ``1 1`` diabat.  Their second differences say
    whether anything in the method kinks where the states cross.
    """
    s_values = payload["s_values"]
    methods = list(payload["scan"])
    confs = ("2 0", "1 1")

    print("\n" + "=" * 78)
    print("configuration-resolved (diabatic) energies through the seam")
    print("=" * 78)

    curves = {}
    for method in methods:
        for conf in confs:
            values = []
            for row in payload["scan"][method]:
                hit = [st for st in row.get("states", [])
                       if st["top_conf"] == conf]
                values.append(hit[0]["energy"] if hit else np.nan)
            curves[(method, conf)] = np.asarray(values, dtype=float)

    header = f"{'s/A':>7s}" + "".join(
        f"{method + ' ' + conf:>20s}" for method in methods for conf in confs
    )
    print(header)
    for i, s in enumerate(s_values):
        line = f"{s:7.4f}"
        for method in methods:
            for conf in confs:
                line += f"{curves[(method, conf)][i]:20.10f}"
        print(line)

    print("\nmax |d2E/ds2| of each diabat (Eh / Angstrom^2):")
    for method in methods:
        for conf in confs:
            d2 = second_difference(curves[(method, conf)], s_values)
            finite = d2[np.isfinite(d2)]
            if finite.size:
                print(f"  {method:10s} {conf}   {np.max(np.abs(finite)):12.6f}")

    if "GBCI" in methods and "CASCI" in methods:
        print("\nGBCI - CASCI per diabat (Eh), i.e. what the grouped bath buys:")
        print(f"{'s/A':>7s} {'2 0':>16s} {'1 1':>16s}")
        for i, s in enumerate(s_values):
            d_closed = curves[("GBCI", "2 0")][i] - curves[("CASCI", "2 0")][i]
            d_open = curves[("GBCI", "1 1")][i] - curves[("CASCI", "1 1")][i]
            print(f"{s:7.4f} {d_closed:16.8f} {d_open:16.8f}")


def second_difference(values, s_values):
    """Discrete second derivative on a possibly non-uniform grid.

    The three-point formula for unequal spacings ``hl`` and ``hr``; NaN at the
    two end points.  Spacings differ between legs of a path whose legs are not
    a whole number of steps long, so the uniform formula would misreport the
    curvature at the junction.
    """
    values = np.asarray(values, dtype=float)
    s_values = np.asarray(s_values, dtype=float)
    out = np.full(values.shape, np.nan)
    hl = s_values[1:-1] - s_values[:-2]
    hr = s_values[2:] - s_values[1:-1]
    out[1:-1] = 2.0 * (
        values[:-2] * hr - values[1:-1] * (hl + hr) + values[2:] * hl
    ) / (hl * hr * (hl + hr))
    return out


def _parse_args(argv):
    """Return ``(path_name, methods, report_only)``."""
    path_name = DEFAULT_PATH
    report_only = False
    methods = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--report":
            report_only = True
        elif token == "--path":
            i += 1
            path_name = argv[i]
        elif token.startswith("--path="):
            path_name = token.split("=", 1)[1]
        else:
            methods.append(token)
        i += 1
    if path_name not in PATHS:
        raise SystemExit(f"unknown path {path_name!r}; choose from {list(PATHS)}")
    for method in methods:
        if method not in METHOD_NROOTS:
            raise SystemExit(f"unknown method {method!r}; choose from {METHODS}")
    return path_name, (methods or list(METHODS)), report_only


def main():
    path_name, requested, report_only = _parse_args(list(sys.argv[1:]))
    run_name = f"{RUN_NAME}_{path_name}"
    output_dir = SCRIPT_DIR / "outputs" / run_name

    if report_only:
        # Re-read a finished scan instead of recomputing it.
        payload = json.loads((output_dir / f"{run_name}.json").read_text())
        report_diabatic(payload)
        return

    spec = PATHS[path_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols, path, waypoints, leg_info, path_curvature = build_path(spec)

    print("===== CH2NH2+ S0/S1 PES smoothness around the MECI =====")
    print(f"basis = {BASIS}, active space = (2e,2o), act_list = {ACT_LIST}")
    print(f"path = {path_name} ({spec.get('mode', 'linear')}), "
          f"{len(path)} points")
    print("waypoints (s = arc length in Angstrom, 0 at "
          f"{spec['origin']}):")
    for wp in waypoints:
        print(f"  s = {wp['s']:+.4f}  {wp['label']:10s} "
              f"(RMSD to {waypoints[0]['label']}: "
              f"{wp['rmsd_to_first_angstrom']:.4f} A)  {wp['file']}")
    for leg in leg_info:
        print(f"  leg {leg['from']} -> {leg['to']}: "
              f"{leg['length_angstrom']:.4f} A in {leg['n_intervals']} steps "
              f"of {leg['step_angstrom']:.4f} A")
    print(f"  extended {spec['extend_angstrom']:.3f} A past both ends")
    if max(path_curvature) > 0.0:
        spread = max(path_curvature) - min(path_curvature)
        detail = ("constant: no point on the path is geometrically special"
                  if spread < 1.0e-8 else
                  "continuous: the curve has no corner")
        print(f"  path curvature |d2R/ds2|: "
              f"{min(path_curvature):.3f} to {max(path_curvature):.3f} 1/A "
              f"({detail})")
    else:
        print("  piecewise linear: each interior waypoint is a corner of the path")

    geometry_rows = []
    for (s, coords), kappa in zip(path, path_curvature):
        geometry_rows.append({
            "s": float(s),
            "path_curvature": float(kappa),
            "r_CN": float(np.linalg.norm(coords[0] - coords[1])),
            "pyr_C": float(pyramidalization_angle(coords, 1, 0, 4, 5)),
            "pyr_N": float(pyramidalization_angle(coords, 0, 1, 2, 3)),
            "torsion_HCNH": float(torsion_angle(coords, 4, 1, 0, 2)),
        })

    print("\ngeometry along the path:")
    print(f"{'s/A':>7s} {'r(CN)/A':>9s} {'pyr(C)/deg':>11s} {'pyr(N)/deg':>11s} "
          f"{'HCNH/deg':>10s}")
    for row in geometry_rows:
        print(f"{row['s']:7.4f} {row['r_CN']:9.4f} {row['pyr_C']:11.3f} "
              f"{row['pyr_N']:11.3f} {row['torsion_HCNH']:10.3f}")

    results = {}
    for method in requested:
        print("\n" + "=" * 78)
        print(f"{method}")
        print("=" * 78)
        rows = []
        for s, coords in path:
            mol = make_mol(symbols, coords)
            scanner = build_scanner(method)  # fresh scanner: no tracking memory
            try:
                row = point_record(method, scanner, mol, s)
            except Exception as exc:
                print(f"  s = {s:+.4f}: FAILED ({exc!r})")
                rows.append({"s": float(s), "method": method, "error": repr(exc)})
                continue
            rows.append(row)
            s0, s1 = row["states"]
            print(f"  s = {s:+.4f}  E0 = {row['E0']:.10f}  E1 = {row['E1']:.10f}  "
                  f"gap = {row['gap_eV']:8.4f} eV  "
                  f"S0[{s0['top_conf']} {s0['top_weight']:.3f}, "
                  f"q(CH2) {s0['q_CH2']:+.3f}]  "
                  f"S1[{s1['top_conf']} {s1['top_weight']:.3f}, "
                  f"q(CH2) {s1['q_CH2']:+.3f}]")
        results[method] = rows

    # Smoothness: second difference of the mean energy along the line.  The
    # conical-intersection cusp lives in the gap, not in the mean, so a spike
    # here is a numerical discontinuity rather than the intersection itself.
    print("\n" + "=" * 78)
    print("smoothness: d2<E>/ds2 of the mean energy (Eh / Angstrom^2)")
    if max(path_curvature) > 0.0:
        print("The path has no corner, so any structure left in these numbers")
        print("belongs to the surfaces rather than to the path.")
    else:
        print("A waypoint where two legs meet is a corner of the path, so every")
        print("method spikes there by the same geometric factor; compare methods.")
    print("=" * 78)
    s_values = [row["s"] for row in geometry_rows]
    waypoint_s = {round(wp["s"], 6): wp["label"] for wp in waypoints}
    header = f"{'s/A':>7s}" + "".join(f"{m:>16s}" for m in requested) + "   waypoint"
    print(header)
    curvature = {}
    for method in requested:
        energies = [row.get("E_mean", np.nan) for row in results[method]]
        curvature[method] = second_difference(energies, s_values)
    for i, (s, _) in enumerate(path):
        line = f"{s:7.4f}"
        for method in requested:
            value = curvature[method][i]
            line += "      nan      " if np.isnan(value) else f"{value:16.6f}"
        label = waypoint_s.get(round(s, 6))
        if label:
            line += f"   <- {label}"
        print(line)

    print("\nmax |d2<E>/ds2| over the interior points:")
    for method in requested:
        finite = curvature[method][np.isfinite(curvature[method])]
        if finite.size:
            print(f"  {method:10s} {np.max(np.abs(finite)):12.6f}")

    payload = {
        "run_name": run_name,
        "path_name": path_name,
        "basis": BASIS,
        "ncas": NCAS,
        "nelecas": list(NELECAS),
        "act_list_1based": ACT_LIST,
        "groupA": GROUPA,
        "method_nroots": METHOD_NROOTS,
        "mode": spec.get("mode", "linear"),
        "waypoints": waypoints,
        "legs": leg_info,
        "path_curvature": path_curvature,
        "step_angstrom": spec["step_angstrom"],
        "extend_angstrom": spec["extend_angstrom"],
        "s_values": s_values,
        "geometry": geometry_rows,
        "scan": results,
        "mean_energy_second_difference": {
            method: [None if np.isnan(x) else float(x) for x in curvature[method]]
            for method in requested
        },
    }
    out_json = output_dir / f"{run_name}.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved: {out_json}")

    report_diabatic(payload)


if __name__ == "__main__":
    main()
