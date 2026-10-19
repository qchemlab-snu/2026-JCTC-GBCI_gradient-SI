#!/bin/bash -i
#
#SBATCH --job-name=3hfref
#SBATCH --output=01_grad_reference_tight.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

# The reference GBCI run behind every finite-difference campaign: tight FASSCF,
# no cache, converged bath CPHF, and the bath semicanonicalized (the default
# since the Newton-SOSCF canonicalization defect was found).  The displaced
# points are NOT recomputed -- the FD energies are invariant to the bath gauge,
# so only the analytic gradient stored in the .npz changes.
python3 -u 01_gbci_s1_grad_check_631g.py --mode reference --no-gbci-cache \
    --fasscf-conv-tol-grad 1e-11 --cphf-tol 1e-16 --cphf-lindep 1e-28 \
    --reference-npz outputs/01_reference_cross_tight_tight.npz
