#!/bin/bash -i
#
#SBATCH --job-name=so2_gbci_grad_check
#SBATCH --output=slurme_so2_gbci_grad_check.out
#SBATCH --ntasks-per-node=40 
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
echo $CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH
# export PYSCF_EXT_PATH=/home/jiseong0626/mygit/grad/pysfnocigrad:$PYSCF_EXT_PATH

# Finite differences amplify BLAS reduction-order noise by 1/(2h); keep this at 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 gbci_so2_grad_check.py > gbci_so2_grad_check.out
