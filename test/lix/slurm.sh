#!/bin/bash -i
#
#SBATCH --job-name=li_halides_gbci_grad_check
#SBATCH --output=slurme_li_halides_gbci_grad_check.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
echo $CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH

# Single-threaded on purpose: threaded BLAS moves the energy by ~1e-9 Eh between
# runs, and the central difference amplifies that by 1/(2h).
export OMP_NUM_THREADS=1

python3 li_halides_gbci_grad_check.py > li_halides_gbci_grad_check.out
python3 li_halides_gbci_grad_fd_sweep.py > li_halides_gbci_grad_fd_sweep.out
#python3 li_halides_orbital_character.py > li_halides_orbital_character.out
