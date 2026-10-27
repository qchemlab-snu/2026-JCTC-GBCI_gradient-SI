# Manuscript and Supporting Information manifest

Where every figure and table in

> Jiseong Park, Minseok Oh and Seunghoon Lee,
> *Analytic Energy Gradient of Grouped-Bath Configuration Interaction (GBCI)*

comes from in this repository, and the command that rebuilds it. Paths are
relative to the repository root.

Every script locates the repository root from its own path, so **run each one
from the directory it lives in** and do not rearrange the tree. The commands
below assume that. Scripts that only read saved results and draw a figure or
write a table need just `numpy`, `matplotlib`, `openpyxl` and `pillow`; the ones
that recompute need the `pyscf-forge` environment described in the README.

## Manuscript

| Item | Figure/table file | Rebuild with | Data behind it |
| --- | --- | --- | --- |
| Table 1 — convergence thresholds | — (typeset by hand) | — | the settings each run recorded: `test/lix/outputs/li_halides_gbci_grad_check.json` (`results[*].calculation`) and `test/3hf/gbci/outputs/01_gbci_s1_grad_check_631g_all_h1e4.json` (`fd_settings`, `reference_run`) |
| Table 2 — LiH/LiF/LiCl analytic vs numerical gradients | — (typeset by hand) | `cd test/lix && python3 li_halides_gbci_grad_fd_sweep.py` | `test/lix/outputs/li_halides_gbci_grad_fd_sweep.{csv,json,xlsx}`, `test/lix/outputs/li_halides_gbci_grad_check.json` |
| Figure 1 — 3-HF structure with atom labels | `test/3hf/gbci/outputs/10_molecule_figure.pdf` | `cd test/3hf/gbci && python3 10_make_molecule_figure.py` | `test/3hf/gbci/outputs/10_atom_labels.json`, `test/3hf/dft/outputs/3hf_dft_S0_opt.xyz` |
| Figure 2 — analytic vs numerical S1 gradient of 3-HF | `test/3hf/gbci/outputs/14_agreement_figure.pdf` | `cd test/3hf/gbci && python3 14_make_agreement_figure.py` | `test/3hf/gbci/outputs/01_gbci_s1_grad_check_631g_all_h1e4.json`, `outputs/10_atom_labels.json` |
| Figure 3 — 4-(1-pyrrolyl)-pyridine structure with atom labels | `test/pp/figures/pp_molecule_figure.pdf` | `cd test/pp/figures && python3 make_molecule_figure.py` | `test/pp/figures/pp_atom_labels.json` |
| Table 3 — PP structural parameters | — (typeset by hand) | — | the same six geometries as Table S10, listed there with their sources |
| Figure 4 — CASCI and GBCI S1 PES | `test/pp/pes/pp_s1_pes_figure.pdf` | `cd test/pp/pes && python3 pp_s1_pes_plot_figure.py` | `test/pp/pes/pp_s0_s1_pes_energies.xlsx` (built by `pp_s0_s1_pes_energies_to_excel.py` from `pp_s1_pes_image_outputs/`, `test/pp/casci/pp_casci_S1_neb_outputs/` and `test/pp/gbci/pp_gbci_S1_planar_tict_neb_outputs/`), plus the three structure renders `S0_casci_S0_GM.png`, `S1_planar_LM.png`, `S1_twisted_LM.png` in the same directory |
| Figure 5 — CH2NH2+ MECI geometries, CAS(2e,2o) | `test/ch2nh2/fig/ch2nh2_meci_active_space_comparison_2o2e.pdf` | `cd test/ch2nh2/fig && python3 make_meci_active_space_comparison_figure.py --active-space 2o2e` | `test/ch2nh2/analysis/outputs/ch2nh2_ccpvtz_meci_descriptors/` and the 215-degree renders in `test/ch2nh2/analysis/outputs/ch2nh2_ccpvtz_meci_images_spin215/` |

## Supporting Information

| Item | Figure/table file | Rebuild with | Data behind it |
| --- | --- | --- | --- |
| Figure S1 — LiH active orbitals | `test/active_orbitals/reference/active_orbital_figures/lih_2e2o/` | see *Active-orbital figures* below | `reference/active_space_moldens/lih_2e2o_active.molden` |
| Figure S2 — LiF active orbitals | `.../lif_4e4o/` | " | `.../lif_4e4o_active.molden` |
| Figure S3 — LiCl active orbitals | `.../licl_4e4o/` | " | `.../licl_4e4o_active.molden` |
| Figure S4 — 3-HF active orbitals | `.../3hf_4e4o/` | " | `.../3hf_4e4o_active.molden` |
| Figure S5 — PP active orbitals | `.../pp_2e2o/` | " | `.../pp_2e2o_active.molden` |
| Figure S6 — CH2NH2+ active orbitals, CAS(2e,2o) | `.../ch2nh2_2e2o/` | " | `.../ch2nh2_2e2o_active.molden`, taken at the GBCI(2e,2o) S0/S1 MECI geometry |
| Tables S1–S3, S5, S6 — configuration groups | `test/bath_groups/outputs/bath_groups.tex` | `cd test/bath_groups && OMP_NUM_THREADS=4 python3 report_bath_groups.py --system LiH --system LiF --system LiCl --system pp --system CH2NH2 --system CH2NH2-2o2e-Lowdin-N --system CH2NH2-2o2e-MO` | `test/bath_groups/outputs/bath_groups.{csv,json}` |
| Table S4 — 3-HF configuration groups | `test/bath_groups/outputs/bath_groups_3hf.tex` | `cd test/bath_groups && OMP_NUM_THREADS=4 python3 report_bath_groups.py --system 3-hydroxyflavone --output outputs/report_bath_groups_3hf.json --latex-output outputs/bath_groups_3hf.tex` | `test/bath_groups/outputs/report_bath_groups_3hf.{csv,json}` |
| Table S7 — 3-HF finite-difference step-size dependence | `test/3hf/gbci/outputs/12_step_size_table.tex` | `cd test/3hf/gbci && python3 12_make_step_size_table.py` | `test/3hf/gbci/outputs/01_gbci_s1_grad_check_631g_all_h{1e4,25e4,5e4,1e3}.json` |
| Table S8 — full 3-HF gradient components | `test/3hf/gbci/outputs/13_full_gradient_table.tex` | `cd test/3hf/gbci && python3 13_make_full_gradient_table.py` | `test/3hf/gbci/outputs/01_gbci_s1_grad_check_631g_all_h1e4.json` |
| Table S9 — 3-HF B3LYP geometry | `test/3hf/gbci/outputs/11_geometry_table.tex` | `cd test/3hf/gbci && python3 11_make_geometry_table.py` | `test/3hf/dft/outputs/3hf_dft_S0_opt.xyz`, `3hf_dft_S0_result.json` |
| Table S10 — PP optimized geometries | `test/supporting_information_geometries.tex` | `cd test/pp/analysis && python3 make_pp_si_geometry_table.py` | the six path endpoints listed in that script: `test/pp/pes/pp_s1_pes_image_outputs/{casci,gbci}_ground_to_planar/..._000.xyz`, `test/pp/casci/pp_casci_S1_neb_outputs/pp_casci_S1_neb_{000,008}.xyz`, `test/pp/gbci/pp_gbci_S1_planar_tict_neb_outputs/..._{000,008}.xyz` |
| Table S11 — CH2NH2+ MECI geometries, CAS(2e,2o) | `test/ch2nh2/analysis/outputs/ch2nh2_ccpvtz_meci_si_tables/ch2nh2_ccpvtz_meci_si_tables_2o2e.tex` | `cd test/ch2nh2/analysis && python3 make_meci_si_geometry_table.py --active-space 2o2e` | `test/ch2nh2/{casci,gbci,casscf}/outputs/` — the runs named in that script's `RUNS` table |
| Figure S7 — SA-CASSCF NEB PES of PP | `test/pp/pes/pp_casscf_s1_neb_pes_figure.pdf` | `cd test/pp/pes && python3 pp_casscf_s1_neb_pes_plot_figure.py` | `test/pp/casscf/pp_casscf_S1_fc_tict_neb_outputs/pp_casscf_S1_fc_tict_neb_profile.csv`, `pp_casscf_neb_{start,end}_structure.png` |
| Figure S8 — SA-CASSCF on the CASCI and GBCI paths | `test/pp/pes/pp_si_casscf_on_casci_neb_figure.pdf` | `cd test/pp/pes && python3 pp_si_casscf_on_casci_neb_plot_figure.py` | `test/pp/casscf/pp_casscf_S1_{pes_ground_to_planar,neb_image,gbci_ground_to_planar,gbci_path}_outputs/` with the matching image directories under `test/pp/pes/pp_s1_pes_image_outputs/`, `test/pp/casci/pp_casci_S1_neb_outputs/` and `test/pp/gbci/pp_gbci_S1_planar_tict_neb_outputs/` |

### Active-orbital figures (Figures S1–S6)

Three steps, all run from `test/active_orbitals/`:

```
python3 export_active_space_moldens.py --systems lih_2e2o,lif_4e4o,licl_4e4o,3hf_4e4o,pp_2e2o,ch2nh2_2e2o
python3 render_active_orbitals.py       # writes the cubes and the per-orbital PNGs
python3 make_active_orbital_panels.py   # assembles each system's panel PDF
```

The moldens are committed; the intermediate `.cube` files are not, because they
are large and `render_active_orbitals.py` regenerates them.

## How the CH2NH2+ (2e,2o) results were produced

The manuscript uses the CAS(2e,2o) space — the C=N pi and pi\* pair, MOs 8 and 9
of the cc-pVTZ RHF reference. The CAS(8e,5o) runs kept alongside them are the
earlier, larger-space results and are what the two-row figure
`make_meci_active_space_comparison_figure.py --active-space all` compares against.

1. CASCI, `test/ch2nh2/casci/casci_2o2e_S0S1_meci_ccpvtz.py`. The published run is
   `--sigma 0.25`: the script's own default of `SIGMA = 1.0` reaches the same
   geometry to 0.0004 A but exhausts `maxsteps` without meeting the gradient
   criterion. The sigma 0.1, 0.5 and 2.0 sweeps are kept beside it.
2. GBCI, `test/ch2nh2/gbci/gbci_2o2e_S0S1_meci_groupocc_ccpvtz.py`, sigma 0.5.
   `gbci_2o2e_S0S1_meci_groupN_ccpvtz.py` is the same active space grouped by
   Loewdin population on nitrogen instead of by orbital occupation; with only two
   active orbitals the two rules give the same three groups, and the two runs
   converge to the same geometry within 8e-07 A. Table S6 reports the
   Loewdin-on-N grouping, which is the one the manuscript describes; Figure 5 and
   Table S11 quote the `groupocc` geometry.
3. SA-CASSCF, `test/ch2nh2/casscf/ch2nh2_casscf_S0S1_meci_2o2e_ccpvtz.py`.
4. `test/ch2nh2/analysis/ch2nh2_meci_geometry_descriptors.py` turns the three
   optimized geometries into the C-N bond lengths and pyramidalization angles
   Figure 5 prints, and writes the aligned XYZ files that
   `make_meci_jmol_images.sh` renders.

The CASCI (2e,2o) run has no penalty Hessian saved, so its MECI is not
frequency-verified; the GBCI and SA-CASSCF (2e,2o) runs are.

## How the Figure S7 path was produced

1. SA-CASSCF S0 and S1 minimizations, `test/pp/casscf/pp_casscf_S0.py` and
   `pp_casscf_S1.py` (driver `pp_casscf_opt.py`), giving the NEB endpoints
   `outputs/pp_casscf_S0/pp_casscf_S0_optimized.xyz` and
   `outputs/pp_casscf_S1/pp_casscf_S1_optimized.xyz`. Both directories hold the
   numerical Hessian and normal modes that confirm each is a minimum.
2. The band itself, `pp_casscf_S1_planar_tict_neb.py` as invoked by
   `slurm_pp_casscf_S1_fc_tict_neb.sh`: nine images seeded from step 36 of the
   earlier `pp_casscf_S1_neb_outputs/pp_casscf_S1_neb.traj`, climbing image off,
   k = 30. Converged at step 10 with a maximum force of 0.0966 eV/Angstrom.
3. `pp_casscf_S1_fc_tict_neb_profile_export.py`, which turns the band into
   `pp_casscf_S1_fc_tict_neb_profile.csv`.
4. The endpoint structures beside the curve: `test/pp/pes/align_neb_endpoints_to_reference.py`
   puts each endpoint in the frame of the published CASCI structure, then
   `render_neb_endpoint_images.sh` with `render_molecule_jmol.spt` renders it.

SA-CASSCF has no planar S1 minimum, which is why this path is separate from the
CASCI and GBCI ones rather than overlaid on them. The supporting runs are
`pp_casscf_S1_from_casci_planar.py` (a minimization started from the planar
structure relaxes all the way to TICT) and the two saddle checks
`pp_casscf_S1_{casci,gbci}_planar_saddle_check.py`, whose summaries record a
gradient far above the convergence threshold and two imaginary modes.
