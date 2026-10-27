#!/bin/bash -i
#
#SBATCH --job-name=opt_pp_S1_twisted
#SBATCH --output=slurme_pp_S1_twisted_init.out
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
export PYSCF_EXT_PATH=/home/jiseong0626/mygit/grad/pysfnocigrad:$PYSCF_EXT_PATH

export OMP_NUM_THREADS=32

python3 pp_casci_S1_twisted_init.py > pp_casci_S1_twisted_init.out
