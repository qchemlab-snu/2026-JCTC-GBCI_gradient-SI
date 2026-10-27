"""Optimize PP S1 with equal-weight S0/S1 SA-CASSCF.

The initial geometry is the near-planar CASCI S1 stationary structure printed
in ``test/pp/casci/pp_casci_S1.out``.  The shared driver also performs the
post-optimization numerical-Hessian frequency check.
"""

from pathlib import Path

import pp_casscf_opt as driver


CASCI_PLANAR_GEOMETRY_SOURCE = "test/pp/casci/pp_casci_S1.out"
CASCI_PLANAR_ATOMS = """C          -3.80391387        0.63517605       -0.03017123
C          -2.51240975        1.26660904       -0.05287748
C          -3.57677148       -0.70316391        0.03024574
C          -2.14932698       -0.87311529        0.05291462
H          -4.74197348        1.16109044       -0.06146135
H          -4.28879337       -1.50911861        0.06155993
N          -1.52638301        0.33335353       -0.00000207
H          -1.59577551       -1.78910122        0.11736982
H          -2.29212559        2.31395580       -0.11731471
C          -0.16296058        0.56468437       -0.00003566
C           0.77715779       -0.50722663       -0.04198334
C           2.11234286       -0.19107093       -0.04198366
H           0.48643252       -1.54474273       -0.09061940
N           2.63104822        1.03869364       -0.00000634
H           2.83293998       -1.00137175       -0.07955565
C           1.73574736        2.02858925        0.04195286
C           0.37102009        1.88667944        0.04193197
H           2.14877567        3.03121151        0.07953387
H          -0.24564392        2.77024895        0.09053472"""


def main() -> None:
    driver.INITIAL_GEOMETRY_SOURCE = CASCI_PLANAR_GEOMETRY_SOURCE
    driver.INITIAL_ATOMS = CASCI_PLANAR_ATOMS

    # Keep this run separate from the existing pp_casscf_S1 calculation.
    default_output_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / "pp_casscf_S1_from_casci_planar"
    )
    original_parse_args = driver.parse_args
    original_optimization_metadata = driver.optimization_metadata

    def parse_args_with_separate_output(default_state=None):
        args = original_parse_args(default_state=default_state)
        if args.output_dir is None:
            args.output_dir = default_output_dir
        return args

    def metadata_for_this_script(*args, **kwargs):
        metadata = original_optimization_metadata(*args, **kwargs)
        metadata["script"] = Path(__file__).name
        return metadata

    driver.parse_args = parse_args_with_separate_output
    driver.optimization_metadata = metadata_for_this_script
    driver.main(default_state="S1")


if __name__ == "__main__":
    main()
