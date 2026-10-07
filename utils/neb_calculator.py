"""
ASE calculator adapters for PySCF gradient scanners.

The main target is NEB calculations driven by ASE while reusing the existing
CASCI active-space/root-tracking scanner in this repository.  ASE expects
energies in eV and forces in eV/Angstrom; the local PySCF scanners return
energies in Hartree and gradients in Eh/Bohr.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pyscf import gto

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:  # pragma: no cover - exercised only when ASE is unavailable.
    Calculator = object  # type: ignore
    all_changes = ("positions", "numbers", "cell", "pbc", "charges", "magmoms")


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
GRADIENT_TO_FORCE_EV_PER_ANGSTROM = -HARTREE_TO_EV / BOHR_TO_ANGSTROM


@dataclass
class ScannerCalculationRecord:
    """Single ASE calculator evaluation backed by a PySCF gradient scanner."""

    energy_hartree: float
    energy_ev: float
    gradient_eh_per_bohr: np.ndarray
    forces_ev_per_angstrom: np.ndarray
    scanner_step: Optional[int]
    converged: Optional[bool]


@dataclass
class NEBRunResult:
    """Summary returned by ``run_neb_calculation``."""

    neb: Any
    images: Sequence[Any]
    optimizer: Any
    barrier: Any
    max_force: Any
    fit: Optional[Any]


def require_ase() -> None:
    """Raise a clear error if ASE is not installed in the active environment."""

    if Calculator is object:
        raise ImportError(
            "ASE is required for NEB calculator support. Install ase in the "
            "active Python environment before running ASE NEB calculations."
        )


def ase_atoms_to_pyscf_atom(ase_atoms: Any) -> Sequence[Tuple[str, Sequence[float]]]:
    """
    Convert an ASE Atoms object to the atom format accepted by PySCF.

    ASE positions are interpreted in Angstrom.  The caller should build the
    PySCF molecule with ``unit='Angstrom'`` unless the positions were prepared
    in another unit intentionally.
    """

    symbols = list(ase_atoms.get_chemical_symbols())
    positions = np.asarray(ase_atoms.get_positions(), dtype=float)

    if positions.shape != (len(symbols), 3):
        raise ValueError(
            "ASE positions must have shape (natm, 3); "
            f"got {positions.shape} for {len(symbols)} atoms."
        )

    return [(symbol, positions[idx].tolist()) for idx, symbol in enumerate(symbols)]


def build_pyscf_mol_from_ase(
    ase_atoms: Any,
    basis: Any,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    verbose: int = 0,
    mol_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build a PySCF molecule from ASE atoms with explicit reproducible settings."""

    kwargs: Dict[str, Any] = dict(mol_kwargs or {})
    kwargs.update(
        {
            "atom": ase_atoms_to_pyscf_atom(ase_atoms),
            "basis": basis,
            "charge": int(charge),
            "spin": int(spin),
            "unit": unit,
            "verbose": int(verbose),
        }
    )
    return gto.M(**kwargs)


def scanner_result_to_energy_gradient(result: Any) -> Tuple[float, np.ndarray]:
    """
    Normalize a scanner return value to ``(energy_hartree, gradient)``.

    The existing single-root CASCI/GBCI trackers return exactly this tuple.
    Keeping the parser small and explicit makes unit conversion failures easier
    to diagnose during NEB runs.
    """

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "Gradient scanner must return (energy_hartree, gradient_eh_per_bohr)."
        )

    energy_hartree, gradient = result
    return float(energy_hartree), np.asarray(gradient, dtype=float)


def gradient_to_ase_forces(gradient_eh_per_bohr: Any) -> np.ndarray:
    """Convert a PySCF nuclear gradient in Eh/Bohr to ASE forces in eV/Angstrom."""

    gradient = np.asarray(gradient_eh_per_bohr, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError(
            "Gradient must have shape (natm, 3) in Eh/Bohr; "
            f"got {gradient.shape}."
        )
    return gradient * GRADIENT_TO_FORCE_EV_PER_ANGSTROM


class PySCFGradientScannerCalculator(Calculator):
    """
    ASE Calculator wrapper for a PySCF scanner returning energy and gradient.

    Parameters
    ----------
    scanner
        Callable receiving a PySCF ``Mole`` and returning
        ``(energy_hartree, gradient_eh_per_bohr)``.
    basis, charge, spin, unit, verbose
        Explicit PySCF molecule settings used for every ASE evaluation.

    Notes
    -----
    For NEB, each image should receive its own calculator and its own scanner
    instance.  The active-space/root-tracking scanners store previous-image
    data internally, so sharing one scanner across images would mix histories.
    """

    name = "PySCFGradientScanner"
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        scanner: Callable[[Any], Tuple[float, np.ndarray]],
        basis: Any,
        charge: int = 0,
        spin: int = 0,
        unit: str = "Angstrom",
        verbose: int = 0,
        mol_kwargs: Optional[Mapping[str, Any]] = None,
        label: Optional[str] = None,
        store_history: bool = True,
        **calculator_kwargs: Any,
    ) -> None:
        require_ase()
        super().__init__(label=label, **calculator_kwargs)

        self.scanner = scanner
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)
        self.unit = unit
        self.verbose = int(verbose)
        self.mol_kwargs = dict(mol_kwargs or {})
        self.store_history = bool(store_history)

        self.last_mol = None
        self.last_energy_hartree: Optional[float] = None
        self.last_gradient_eh_per_bohr: Optional[np.ndarray] = None
        self.last_record: Optional[ScannerCalculationRecord] = None
        self.history = []

    def make_mol(self, atoms: Any) -> Any:
        """Build the PySCF molecule used for the current ASE image."""

        return build_pyscf_mol_from_ase(
            ase_atoms=atoms,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            unit=self.unit,
            verbose=self.verbose,
            mol_kwargs=self.mol_kwargs,
        )

    def calculate(
        self,
        atoms: Any = None,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        """Evaluate energy and forces for ASE."""

        require_ase()
        super().calculate(atoms, properties, system_changes)

        mol = self.make_mol(self.atoms)
        energy_hartree, gradient = scanner_result_to_energy_gradient(
            self.scanner(mol)
        )
        forces = gradient_to_ase_forces(gradient)
        energy_ev = energy_hartree * HARTREE_TO_EV

        if forces.shape[0] != len(self.atoms):
            raise ValueError(
                "Scanner gradient atom count does not match ASE atoms: "
                f"{forces.shape[0]} != {len(self.atoms)}."
            )

        scanner_step = getattr(self.scanner, "step", None)
        converged = getattr(self.scanner, "converged", None)
        if converged is not None:
            converged = bool(converged)

        record = ScannerCalculationRecord(
            energy_hartree=energy_hartree,
            energy_ev=energy_ev,
            gradient_eh_per_bohr=gradient.copy(),
            forces_ev_per_angstrom=forces.copy(),
            scanner_step=scanner_step,
            converged=converged,
        )

        self.results["energy"] = energy_ev
        self.results["forces"] = forces

        self.last_mol = mol
        self.last_energy_hartree = energy_hartree
        self.last_gradient_eh_per_bohr = gradient.copy()
        self.last_record = record
        if self.store_history:
            self.history.append(record)


class CASCIActiveSpaceNEBCalculator(PySCFGradientScannerCalculator):
    """ASE NEB calculator for ``CASCI_Active_Root_Tracking_Optimizer`` scanners."""

    name = "CASCIActiveSpaceNEB"


class CASSCFActiveSpaceNEBCalculator(PySCFGradientScannerCalculator):
    """ASE NEB calculator for ``CASSCF_Active_Root_Tracking_Optimizer`` scanners."""

    name = "CASSCFActiveSpaceNEB"


def make_gradient_scanner_calculator_factory(
    scanner_factory: Callable[[], Callable[[Any], Tuple[float, np.ndarray]]],
    basis: Any,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    verbose: int = 0,
    mol_kwargs: Optional[Mapping[str, Any]] = None,
    label_prefix: Optional[str] = None,
    store_history: bool = True,
    calculator_class: Callable[..., PySCFGradientScannerCalculator] = (
        PySCFGradientScannerCalculator
    ),
    **calculator_kwargs: Any,
) -> Callable[..., PySCFGradientScannerCalculator]:
    """
    Return a factory that creates fresh scanner calculators for NEB images.

    The returned callable accepts an optional ``label`` keyword.  If no label is
    given, labels are generated from ``label_prefix`` and an image counter.
    """

    counter = {"index": 0}

    def make_calculator(label: Optional[str] = None) -> PySCFGradientScannerCalculator:
        idx = counter["index"]
        counter["index"] += 1
        if label is None and label_prefix is not None:
            label_value = f"{label_prefix}_{idx:03d}"
        else:
            label_value = label

        return calculator_class(
            scanner=scanner_factory(),
            basis=basis,
            charge=charge,
            spin=spin,
            unit=unit,
            verbose=verbose,
            mol_kwargs=mol_kwargs,
            label=label_value,
            store_history=store_history,
            **calculator_kwargs,
        )

    return make_calculator


def make_casci_active_space_calculator_factory(
    scanner_kwargs: Mapping[str, Any],
    basis: Any,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    verbose: int = 0,
    mol_kwargs: Optional[Mapping[str, Any]] = None,
    label_prefix: Optional[str] = "casci_neb_image",
    store_history: bool = True,
    **calculator_kwargs: Any,
) -> Callable[..., CASCIActiveSpaceNEBCalculator]:
    """
    Build a factory for CASCI active-space tracked ASE NEB calculators.

    Example
    -------
    ``factory = make_casci_active_space_calculator_factory(...)``
    followed by ``image.calc = factory()`` for each NEB image.
    """

    scanner_config = dict(scanner_kwargs)

    def scanner_factory() -> Any:
        from utils.active_space_tracking_optimizer import (
            CASCI_Active_Root_Tracking_Optimizer,
        )

        return CASCI_Active_Root_Tracking_Optimizer(**scanner_config)

    return make_gradient_scanner_calculator_factory(
        scanner_factory=scanner_factory,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        verbose=verbose,
        mol_kwargs=mol_kwargs,
        label_prefix=label_prefix,
        store_history=store_history,
        calculator_class=CASCIActiveSpaceNEBCalculator,
        **calculator_kwargs,
    )


def make_casscf_active_space_calculator_factory(
    scanner_kwargs: Mapping[str, Any],
    basis: Any,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    verbose: int = 0,
    mol_kwargs: Optional[Mapping[str, Any]] = None,
    label_prefix: Optional[str] = "casscf_neb_image",
    store_history: bool = True,
    **calculator_kwargs: Any,
) -> Callable[..., CASSCFActiveSpaceNEBCalculator]:
    """
    Build a factory for CASSCF active-space tracked ASE NEB calculators.

    Mirrors :func:`make_casci_active_space_calculator_factory`, but builds
    ``CASSCF_Active_Root_Tracking_Optimizer`` scanners, so ``scanner_kwargs``
    also accepts the state-averaging options (``state_average``,
    ``state_weights``, ``project_previous_orbitals``, ``reuse_ci_guess``).

    Each NEB image gets its own scanner, so orbital and CI guesses are carried
    along that image's own history rather than shared between images.
    """

    scanner_config = dict(scanner_kwargs)

    def scanner_factory() -> Any:
        from utils.active_space_tracking_optimizer import (
            CASSCF_Active_Root_Tracking_Optimizer,
        )

        return CASSCF_Active_Root_Tracking_Optimizer(**scanner_config)

    return make_gradient_scanner_calculator_factory(
        scanner_factory=scanner_factory,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        verbose=verbose,
        mol_kwargs=mol_kwargs,
        label_prefix=label_prefix,
        store_history=store_history,
        calculator_class=CASSCFActiveSpaceNEBCalculator,
        **calculator_kwargs,
    )


def attach_calculators_to_images(
    images: Sequence[Any],
    calculator_factory: Callable[[], Any],
) -> Sequence[Any]:
    """Attach one freshly created ASE calculator to each NEB image."""

    for image in images:
        image.calc = calculator_factory()
    return images


def build_neb_images(
    initial: Any,
    final: Any,
    n_intermediate_images: int,
    calculator_factory: Optional[Callable[[], Any]] = None,
    copy_endpoints: bool = True,
) -> List[Any]:
    """
    Build an ASE NEB image list from endpoint geometries.

    ``n_intermediate_images`` counts only the images between the endpoints.
    If ``calculator_factory`` is provided, each image receives a separate
    calculator instance.
    """

    if n_intermediate_images < 0:
        raise ValueError("n_intermediate_images must be non-negative.")

    initial_image = initial.copy() if copy_endpoints else initial
    final_image = final.copy() if copy_endpoints else final
    images = [initial_image]
    images.extend(initial.copy() for _ in range(int(n_intermediate_images)))
    images.append(final_image)

    if calculator_factory is not None:
        attach_calculators_to_images(images, calculator_factory)

    return images


def get_ase_neb_classes() -> Tuple[Any, Any]:
    """Import ASE NEB classes with compatibility for old and new ASE layouts."""

    require_ase()
    try:
        from ase.mep import NEB, NEBTools
    except ImportError:
        from ase.neb import NEB, NEBTools
    return NEB, NEBTools


def get_ase_optimizer_class(optimizer: Any) -> Any:
    """Resolve an ASE optimizer name or return a user-provided optimizer class."""

    require_ase()
    if not isinstance(optimizer, str):
        return optimizer

    name = optimizer.upper()
    if name == "FIRE":
        from ase.optimize import FIRE

        return FIRE
    if name == "LBFGS":
        from ase.optimize import LBFGS

        return LBFGS

    raise ValueError("optimizer must be 'FIRE', 'LBFGS', or an optimizer class.")


def run_neb_calculation(
    images: Sequence[Any],
    climb: bool = True,
    k: float = 0.1,
    interpolate: Optional[str] = "idpp",
    optimizer: Any = "FIRE",
    fmax: float = 0.05,
    steps: Optional[int] = None,
    trajectory: Optional[Any] = None,
    logfile: Optional[Any] = None,
    neb_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
) -> NEBRunResult:
    """
    Run an ASE NEB calculation for images that already have calculators.

    This is intentionally a thin ASE wrapper: the PySCF/CASCI behavior stays in
    each image calculator, while ASE owns interpolation, NEB forces, and the
    optimizer.
    """

    NEB, NEBTools = get_ase_neb_classes()
    neb = NEB(images, climb=bool(climb), k=float(k), **dict(neb_kwargs or {}))

    if interpolate is not None:
        neb.interpolate(interpolate)

    optimizer_class = get_ase_optimizer_class(optimizer)
    opt_kwargs = dict(optimizer_kwargs or {})
    if trajectory is not None:
        opt_kwargs["trajectory"] = str(trajectory)
    if logfile is not None:
        opt_kwargs["logfile"] = str(logfile)

    opt = optimizer_class(neb, **opt_kwargs)
    run_kwargs: Dict[str, Any] = {"fmax": float(fmax)}
    if steps is not None:
        run_kwargs["steps"] = int(steps)
    opt.run(**run_kwargs)

    tools = NEBTools(images)
    try:
        fit = tools.get_fit()
    except Exception:
        fit = None

    return NEBRunResult(
        neb=neb,
        images=images,
        optimizer=opt,
        barrier=tools.get_barrier(),
        max_force=tools.get_fmax(),
        fit=fit,
    )


def write_neb_images(
    images: Sequence[Any],
    output_dir: Any,
    prefix: str = "neb_image",
) -> List[Path]:
    """Write NEB images as XYZ files and return the written paths."""

    require_ase()
    from ase import io

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = []
    for idx, image in enumerate(images):
        path = output_path / f"{prefix}_{idx:03d}.xyz"
        io.write(str(path), image)
        written.append(path)
    return written
