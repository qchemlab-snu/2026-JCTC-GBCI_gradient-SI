#!/bin/bash -i
#
#SBATCH --job-name=3hffcmbh1e4
#SBATCH --output=slurme_3hf_combine_tight_frozen_h1e4.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -u 01_gbci_s1_grad_check_631g.py --mode combine --step-bohr 1e-4 \
    --reference-npz outputs/01_reference_cross_tight_tight.npz \
    --partial-dir outputs/01_partials_tight_frozen_h1e4 \
    --output outputs/01_gbci_s1_grad_check_631g_tight_frozen_h1e4.json \
    > 01_grad_combine_tight_frozen_h1e4.out
