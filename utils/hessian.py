from pathlib import Path

import numpy as np

def numerical_hessian_from_gradient_scanner(scanner, mol, step=1e-3):
    """
    scanner(mol) must return:
        energy, gradient

    gradient shape:
        (natm, 3)

    step:
        displacement in Bohr

    returns:
        hess4: shape (natm, natm, 3, 3)
        hess2: shape (3*natm, 3*natm)
    """

    mol0 = mol.copy()
    coords0 = mol0.atom_coords(unit="Bohr")
    natm = mol0.natm

    hess4 = np.zeros((natm, natm, 3, 3))

    for ia in range(natm):
        for ix in range(3):
            coords_p = coords0.copy()
            coords_m = coords0.copy()

            coords_p[ia, ix] += step
            coords_m[ia, ix] -= step

            mol_p = mol0.copy()
            mol_m = mol0.copy()

            mol_p.set_geom_(coords_p, unit="Bohr")
            mol_m.set_geom_(coords_m, unit="Bohr")

            _, grad_p = scanner(mol_p)
            _, grad_m = scanner(mol_m)

            # derivative of all gradient components wrt coordinate (ia, ix)
            dgrad = (np.asarray(grad_p) - np.asarray(grad_m)) / (2.0 * step)

            # hess4[displaced_atom, response_atom, displaced_xyz, response_xyz]
            hess4[ia, :, ix, :] = dgrad

    # Convert to conventional 2D Cartesian Hessian:
    # index = 3*atom + xyz
    hess2 = hess4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    # Symmetrize to reduce numerical noise
    hess2 = 0.5 * (hess2 + hess2.T)

    return hess4, hess2


def numerical_hessian_from_scanner_factory(
    scanner_factory,
    mol,
    step=1e-3,
    cache_dir=None,
    cache_key=None,
):
    """
    Build a numerical Hessian with independent scanners for every displacement.

    ``scanner_factory`` must return a new callable that accepts a molecule and
    returns ``(energy, gradient)``.  This avoids path-dependent state changes in
    active-space/root-tracking scanners.  If ``cache_dir`` is provided, each
    displaced gradient is saved separately and reused on restart.  ``cache_key``
    can identify the electronic-structure reference so that stale gradients
    from a different wavefunction are rejected.
    """
    step = float(step)
    if step <= 0:
        raise ValueError("step must be positive.")

    mol0 = mol.copy()
    coords0 = mol0.atom_coords(unit="Bohr")
    natm = mol0.natm
    expected_gradient_shape = (natm, 3)
    cache_path = None if cache_dir is None else Path(cache_dir)
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    def evaluate_displacement(atom_index, axis_index, sign):
        coords = coords0.copy()
        coords[atom_index, axis_index] += sign * step

        displaced_mol = mol0.copy()
        displaced_mol.set_geom_(coords, unit="Bohr")

        cache_file = None
        if cache_path is not None:
            sign_label = "plus" if sign > 0 else "minus"
            cache_file = cache_path / (
                f"atom_{atom_index:03d}_axis_{axis_index}_{sign_label}.npz"
            )

        if cache_file is not None and cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as saved:
                saved_coords = np.asarray(saved["coords_bohr"], dtype=float)
                saved_step = float(saved["step_bohr"])
                gradient = np.asarray(saved["gradient"], dtype=float)
                saved_cache_key = str(saved["cache_key"])

            if not np.allclose(saved_coords, coords, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Cached geometry does not match the requested displacement: "
                    f"{cache_file}"
                )
            if abs(saved_step - step) > 1e-14:
                raise ValueError(
                    f"Cached Hessian step does not match step={step}: {cache_file}"
                )
            if gradient.shape != expected_gradient_shape:
                raise ValueError(
                    f"Cached gradient has shape {gradient.shape}; expected "
                    f"{expected_gradient_shape}: {cache_file}"
                )
            if cache_key is not None and saved_cache_key != str(cache_key):
                raise ValueError(
                    f"Cached electronic-structure reference does not match: "
                    f"{cache_file}"
                )

            print(f"Loaded displaced gradient: {cache_file}")
            return gradient

        scanner = scanner_factory()
        energy, gradient = scanner(displaced_mol)
        gradient = np.asarray(gradient, dtype=float)

        if gradient.shape != expected_gradient_shape:
            raise ValueError(
                f"Gradient has shape {gradient.shape}; expected "
                f"{expected_gradient_shape}."
            )
        if not np.all(np.isfinite(gradient)):
            raise ValueError("Displaced gradient contains non-finite values.")

        if cache_file is not None:
            temporary_file = cache_file.with_suffix(".tmp.npz")
            np.savez(
                temporary_file,
                coords_bohr=coords,
                step_bohr=step,
                energy_hartree=float(energy),
                gradient=gradient,
                cache_key="" if cache_key is None else str(cache_key),
            )
            temporary_file.replace(cache_file)
            print(f"Saved displaced gradient: {cache_file}")

        return gradient

    hess4 = np.zeros((natm, natm, 3, 3))
    for atom_index in range(natm):
        for axis_index in range(3):
            grad_plus = evaluate_displacement(atom_index, axis_index, +1)
            grad_minus = evaluate_displacement(atom_index, axis_index, -1)
            hess4[atom_index, :, axis_index, :] = (
                grad_plus - grad_minus
            ) / (2.0 * step)

    hess2 = hess4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    hess2 = 0.5 * (hess2 + hess2.T)
    return hess4, hess2

AMU2ME = 1822.888486209
BOHR2ANG = 0.529177210903
ANG2BOHR = 1.0 / BOHR2ANG
AU_FREQ_TO_CM = 219474.6313705


def get_masses_amu(mol):
    try:
        return np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)
    except TypeError:
        return np.asarray(mol.atom_mass_list(), dtype=float)


def imaginary_mode_to_displaced_mols(
    mol,
    hess2,
    imag_threshold_cm=30.0,
    amplitude=0.05,
    amplitude_unit="Angstrom",
    save_path=None,
):
    """
    Check frequencies from a Cartesian Hessian and, if an imaginary mode exists,
    generate plus/minus displaced PySCF Mole objects along the first imaginary mode.

    Returns
    -------
    freqs_cm:
        All frequencies in cm^-1.
        Real frequencies are positive.
        Imaginary frequencies are returned as negative values.

    modes_cart:
        Cartesian normal modes with shape (3N, natm, 3).
        Each mode is normalized so that the largest atomic displacement norm is 1.
        Unitless direction vector in Cartesian coordinates.

    imag_freq_cm:
        Frequency of the selected imaginary mode in cm^-1.
        If no imaginary frequency is found, returns None.

    imag_mode_cart:
        Cartesian displacement direction of the selected imaginary mode.
        Shape is (natm, 3).
        Normalized so that max atomic displacement norm is 1.
        If no imaginary frequency is found, returns None.

    mol_plus:
        Mole displaced along the + imaginary mode.
        If no imaginary frequency is found, returns None.

    mol_minus:
        Mole displaced along the - imaginary mode.
        If no imaginary frequency is found, returns None.
    """

    natm = mol.natm
    hess2 = np.asarray(hess2, dtype=float)

    if hess2.shape != (3 * natm, 3 * natm):
        raise ValueError(
            f"hess2 must have shape ({3 * natm}, {3 * natm}), "
            f"but got {hess2.shape}."
        )

    # Symmetrize Hessian
    hess2 = 0.5 * (hess2 + hess2.T)

    # Atomic masses in electron-mass units
    masses_amu = get_masses_amu(mol)
    masses_au = masses_amu * AMU2ME
    masses_au_cart = np.repeat(masses_au, 3)

    # Mass-weighted Hessian
    hess_mw = hess2 / np.sqrt(np.outer(masses_au_cart, masses_au_cart))
    hess_mw = 0.5 * (hess_mw + hess_mw.T)

    eigvals, eigvecs_mw = np.linalg.eigh(hess_mw)

    # Signed frequencies in cm^-1
    # Negative values correspond to imaginary frequencies.
    freqs_cm = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * AU_FREQ_TO_CM

    # Convert all mass-weighted normal modes to Cartesian displacement directions
    modes_cart = []
    for imode in range(3 * natm):
        mode_mw = eigvecs_mw[:, imode]
        mode_cart = mode_mw / np.sqrt(masses_au_cart)
        mode_cart = mode_cart.reshape(natm, 3)

        atom_norms = np.linalg.norm(mode_cart, axis=1)
        max_norm = np.max(atom_norms)

        if max_norm > 1e-14:
            mode_cart = mode_cart / max_norm

        modes_cart.append(mode_cart)

    modes_cart = np.asarray(modes_cart)  # shape: (3N, natm, 3)

    imag_indices = np.where(freqs_cm < -abs(imag_threshold_cm))[0]

    if len(imag_indices) == 0:
        if save_path is not None:
            np.savez(
                save_path,
                freqs_cm=freqs_cm,
                modes_cart=modes_cart,
                imag_mode_index=None,
                imag_freq_cm=None,
                imag_mode_cart=None,
            )

        return freqs_cm, modes_cart, None, None, None, None

    # Use the most negative frequency mode
    mode_idx = imag_indices[0]

    imag_freq_cm = freqs_cm[mode_idx]
    imag_mode_cart = modes_cart[mode_idx]

    # Convert amplitude to Bohr
    if amplitude_unit.lower().startswith("ang"):
        amp_bohr = amplitude * ANG2BOHR
    elif amplitude_unit.lower().startswith("bohr"):
        amp_bohr = amplitude
    else:
        raise ValueError("amplitude_unit must be 'Angstrom' or 'Bohr'.")

    coords0 = mol.atom_coords(unit="Bohr")
    disp = amp_bohr * imag_mode_cart

    mol_plus = mol.copy()
    mol_minus = mol.copy()

    mol_plus.set_geom_(coords0 + disp, unit="Bohr", inplace=True)
    mol_minus.set_geom_(coords0 - disp, unit="Bohr", inplace=True)

    if save_path is not None:
        np.savez(
            save_path,
            freqs_cm=freqs_cm,
            modes_cart=modes_cart,
            imag_mode_index=mode_idx,
            imag_freq_cm=imag_freq_cm,
            imag_mode_cart=imag_mode_cart,
        )

    return freqs_cm, modes_cart, imag_freq_cm, imag_mode_cart, mol_plus, mol_minus

def displaced_mols_from_saved_normal_mode(
    mol,
    mode_file,
    mode_index,
    amplitude=0.05,
    amplitude_unit="Angstrom",
    renormalize=True,
):
    """
    Load saved normal modes from .npz file and generate plus/minus displaced
    PySCF Mole objects along a selected normal mode.

    Parameters
    ----------
    mol:
        Reference PySCF Mole object.
        The displacement is applied to this geometry.

    mode_file:
        Path to .npz file saved by imaginary_mode_to_displaced_mols.
        It should contain:
            freqs_cm
            modes_cart

    mode_index:
        Index of the normal mode to use.
        This follows the order of np.linalg.eigh eigenvalues.
        Usually modes are sorted from lowest frequency to highest frequency.
        Python-style zero-based index.

    amplitude:
        Maximum atomic displacement along the selected mode.

    amplitude_unit:
        "Angstrom" or "Bohr".

    renormalize:
        If True, renormalize the loaded mode so that
            max_i |dR_i| = 1
        before applying amplitude.

    Returns
    -------
    freq_cm:
        Frequency of the selected mode in cm^-1.

    mode_cart:
        Cartesian displacement direction with shape (natm, 3).

    mol_plus:
        Mole displaced along +mode.

    mol_minus:
        Mole displaced along -mode.
    """

    data = np.load(mode_file, allow_pickle=True)

    if "freqs_cm" not in data:
        raise KeyError(f"{mode_file} does not contain 'freqs_cm'.")

    if "modes_cart" not in data:
        raise KeyError(f"{mode_file} does not contain 'modes_cart'.")

    freqs_cm = data["freqs_cm"]
    modes_cart = data["modes_cart"]

    natm = mol.natm
    nmode = 3 * natm

    if modes_cart.shape != (nmode, natm, 3):
        raise ValueError(
            f"modes_cart must have shape ({nmode}, {natm}, 3), "
            f"but got {modes_cart.shape}."
        )

    if mode_index < 0 or mode_index >= nmode:
        raise ValueError(
            f"mode_index must be between 0 and {nmode - 1}, "
            f"but got {mode_index}."
        )

    freq_cm = freqs_cm[mode_index]
    mode_cart = np.array(modes_cart[mode_index], dtype=float, copy=True)

    if renormalize:
        atom_norms = np.linalg.norm(mode_cart, axis=1)
        max_norm = np.max(atom_norms)

        if max_norm < 1e-14:
            raise ValueError(
                f"Normal mode {mode_index} is numerically zero."
            )

        mode_cart /= max_norm

    # Convert amplitude to Bohr
    if amplitude_unit.lower().startswith("ang"):
        amp_bohr = amplitude * ANG2BOHR
    elif amplitude_unit.lower().startswith("bohr"):
        amp_bohr = amplitude
    else:
        raise ValueError("amplitude_unit must be 'Angstrom' or 'Bohr'.")

    coords0 = mol.atom_coords(unit="Bohr")
    disp = amp_bohr * mode_cart

    mol_plus = mol.copy()
    mol_minus = mol.copy()

    mol_plus.set_geom_(coords0 + disp, unit="Bohr", inplace=True)
    mol_minus.set_geom_(coords0 - disp, unit="Bohr", inplace=True)

    return freq_cm, mode_cart, mol_plus, mol_minus
