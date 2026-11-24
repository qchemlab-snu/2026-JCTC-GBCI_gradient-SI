#!/bin/bash -i
#
#SBATCH --job-name=3hfref02
#SBATCH --output=01_grad_reference_thr02.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

# Seven bath groups, against the four of the earlier 0.5 setting.  The
# complete-linkage partition is unchanged for every threshold in
# [0.1559, 0.2705) -- both edges located by bisection -- so 0.2 and 0.16 give
# the same seven groups and, with the grouping frozen, identical energies.
# 0.2 is the value the manuscript quotes.  This grouping is what the displaced
# points are frozen to, so it has to finish before the finite differences.
python3 -u 01_gbci_s1_grad_check_631g.py --mode reference --no-gbci-cache \
    --group-threshold 0.2 \
    --fasscf-conv-tol-grad 1e-11 --cphf-tol 1e-16 --cphf-lindep 1e-28 \
    --reference-npz outputs/01_reference_thr02.npz
