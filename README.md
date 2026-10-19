# GBCI-grad
Analytic energy gradient for Grouped-Bath Configuration Interaction (GBCI)

## Requirements

This repository was developed and tested with the
[qchemlab-snu pyscf-forge fork](https://github.com/qchemlab-snu/pyscf-forge),
which supplies the GBCI wave-function code (`pyscf.gbci`) that the gradient
module builds on.

`pyscf/grad/gbci.py` in this repository is the analytic-gradient implementation
that produced the published results, including the 2026-09-30 corrections. 

The calculation and geometry-optimization scripts also require
[geomeTRIC](https://github.com/leeping/geomeTRIC) and
[ASE](https://wiki.fysik.dtu.dk/ase/). 

## Repository layout

```text
GBCI-grad/
├── pyscf/
│   └── grad/
│       └── gbci.py                    # Analytic GBCI nuclear-gradient implementation
├── utils/
│   ├── active_space_tracking_optimizer.py
│   │                                  # CASCI/CASSCF/GBCI active-space and root tracking
│   ├── bath_grouping.py               # Complete-linkage grouping of bath configurations
│   ├── bath_semicanonical.py          # Semicanonicalizes the group bath orbitals
│   ├── gbci_compat.py                 # Legacy input conversion for the fork's GBCI API
│   ├── gbci_setup.py                  # Shared GBCI construction for the gradient checks
│   ├── geometry_descriptors.py        # Bond lengths, pyramidalization and XYZ alignment
│   ├── hessian.py                     # Numerical Hessian and normal-mode helpers
│   ├── meci_optimizer.py              # Two-state tracking and penalty-function MECI tools
│   └── neb_calculator.py              # ASE calculator factories and NEB helpers
├── test/
│   ├── lix/                           # LiH/LiF/LiCl analytic-vs-finite-difference checks
│   │   └── outputs/                   # Saved sweep tables and JSON results
│   ├── bath_groups/                   # Configuration-grouping tables for every system
│   │   └── outputs/                   # bath_groups.tex as included in the Supporting Information
│   ├── 3hf/                           # 3-hydroxyflavone S1 analytic-vs-finite-difference checks
│   │   ├── dft/                       # B3LYP/6-31G(d) ground-state optimization
│   │   └── gbci/                      # GBCI(4e,4o)/6-31G S1 gradients, step sweep, tables and figure
│   │       └── outputs/01_partials*/  # Per-task finite-difference energies behind each step size
│   ├── active_orbitals/               # Active-space orbital export and figure rendering
│   │   └── reference/                 # Active-space moldens and rendered orbital panels
│   ├── ch2nh2/                        # Protonated formaldimine, CAS(8e,5o)/cc-pVTZ
│   │   ├── casci/                     # CASCI S0/S1 MECI workflow and saved results
│   │   ├── gbci/                      # GBCI S0/S1 MECI workflow and saved results
│   │   ├── casscf/                    # SA-CASSCF S0/S1 MECI workflow and saved results
│   │   ├── analysis/                  # Geometry descriptors, MECI images and the SI geometry table
│   │   └── fig/                       # Three-panel MECI geometry comparison figure
│   └── pp/
│       ├── casci/                     # 4-(1-pyrrolyl)-pyridine CASCI optimizations and NEB calculations
│       ├── gbci/                      # 4-(1-pyrrolyl)-pyridine GBCI optimizations and NEB calculations
│       ├── casscf/                    # SA-CASSCF single points on the CASCI and GBCI paths
│       ├── figures/                   # Structure figure with the manuscript's atom labeling
│       └── pes/                       # 4-(1-pyrrolyl)-pyridine potential-energy-path construction and export
├── .gitignore
└── README.md
```

`pyscf/grad/gbci.py` mirrors the `pyscf.grad` namespace and is intended to be
loaded as the GBCI gradient module in a PySCF/pyscf-forge environment. The
`utils/` modules are shared by the calculation scripts under `test/`.

Two of the validation scripts — `test/lix/li_halides_gbci_grad_check.py` and
`test/3hf/gbci/01_gbci_s1_grad_check_631g.py` — refuse to run when
`pyscf.grad.gbci` resolves to a copy inside this repository, because they were
written to measure whichever module `PYSCF_EXT_PATH` selects. To run them, place
`pyscf/grad/gbci.py` on the `pyscf-forge` extension path rather than importing it
from the repository root.

The `test/` tree contains reproducible research calculations and their archived
outputs rather than only lightweight unit tests. Output directories may include
optimized XYZ geometries, JSON/CSV metadata, Excel
summaries, plots, trajectories, and calculation logs.

Every calculation script locates the repository root from its own path, so scripts
must be run from the directory they live in and the tree must not be rearranged.

`test/active_orbitals/reference/active_orbital_figures/` ships the rendered orbital
panels but not the intermediate `.cube` files, which are large and are regenerated
from the moldens in `active_space_moldens/` by `render_active_orbitals.py`.

Calculation scripts whose names end in `_im1.py` or `_im2.py` are follow-up
optimizations for an imaginary-frequency mode. They start from geometries
displaced in the plus and minus directions, respectively, along the imaginary
normal mode and then optimize each displaced geometry again.
