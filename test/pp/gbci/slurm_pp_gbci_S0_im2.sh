#!/bin/bash -i
#
#SBATCH --job-name=opt_pp_S0_im2
#SBATCH --output=test/pp/gbci/slurme_pp_gbci_S0_im2.out
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 32core_partition

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH
export PYSCF_EXT_PATH=$PWD:$PYSCF_EXT_PATH
export OMP_NUM_THREADS=32

python3 test/pp/gbci/pp_gbci_S0_im2.py > test/pp/gbci/pp_gbci_S0_im2.out
