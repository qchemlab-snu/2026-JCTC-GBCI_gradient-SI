#!/bin/bash -i
#
#SBATCH --job-name=3hf68
#SBATCH --output=logs_thr02/gesvd68_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# gesdd (numpy's default SVD driver) fails deterministically on this one
# displaced geometry's biorthogonalization overlap; 01_run_with_gesvd.py
# retries with the QR-based gesvd driver and is inert everywhere else.
python3 -u 01_run_with_gesvd.py 01_gbci_s1_grad_check_631g.py \
    --mode fd --step-bohr 1e-3 --fd-atoms all --freeze-groups \
    --no-gbci-cache --group-threshold 0.2 --fasscf-conv-tol-grad 1e-11 \
    --verbose 3 --reference-npz outputs/01_reference_thr02.npz \
    --partial-dir outputs/01_partials_thr02_h1e3 \
    --task-id 68 --num-tasks 84
