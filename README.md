# GBCI-grad
Analytic energy gradient for Grouped-Bath Configuration Interaction (GBCI)

## Requirements

This repository was developed and tested with the
[qchemlab-snu pyscf-forge fork](https://github.com/qchemlab-snu/pyscf-forge),
including its GBCI implementation. The compatibility baseline used here is the
fork's `master` commit `4e014735` (2026-08-11).

The calculation and geometry-optimization scripts also require
[geomeTRIC](https://github.com/leeping/geomeTRIC) and
[ASE](https://wiki.fysik.dtu.dk/ase/). Install the Python packages with, for
example:

```bash
python -m pip install geometric ase
```

## Repository layout

```text
GBCI-grad/
├── pyscf/
│   └── grad/
│       └── gbci.py                    # Analytic GBCI nuclear-gradient implementation
├── utils/
│   ├── active_space_tracking_optimizer.py
│   │                                  # CASCI/GBCI active-space and root tracking
│   ├── gbci_compat.py                 # Legacy input conversion for the fork's GBCI API
│   ├── hessian.py                     # Numerical Hessian and normal-mode helpers
│   ├── meci_optimizer.py              # Two-state tracking and penalty-function MECI tools
│   └── neb_calculator.py              # ASE calculator factories and NEB helpers
├── test/
│   ├── lix/                           # LiH/LiF/LiCl analytic-vs-finite-difference checks
│   │   └── outputs/                   # Saved sweep tables and JSON results
│   ├── ch2nh2/
│   │   ├── casci/                     # CH2NH2 CASCI MECI workflow and saved results
│   │   └── gbci/                      # CH2NH2 GBCI optimization workflow and saved results
│   └── pp/
│       ├── casci/                     # PP CASCI optimizations and NEB calculations
│       ├── gbci/                      # PP GBCI optimizations and NEB calculations
│       └── pes/                       # PP potential-energy-path construction and export
├── .gitignore
└── README.md
```

`pyscf/grad/gbci.py` mirrors the `pyscf.grad` namespace and is intended to be
loaded as the GBCI gradient module in a PySCF/pyscf-forge environment. The
`utils/` modules are shared by the calculation scripts under `test/`.

The `test/` tree contains reproducible research calculations and their archived
outputs rather than only lightweight unit tests. Output directories may include
optimized XYZ geometries, NumPy archives (`.npz`), JSON/CSV metadata, Excel
summaries, Molden orbital files, plots, trajectories, and calculation logs.

Calculation scripts whose names end in `_im1.py` or `_im2.py` are follow-up
optimizations for an imaginary-frequency mode. They start from geometries
displaced in the plus and minus directions, respectively, along the imaginary
normal mode and then optimize each displaced geometry again.
