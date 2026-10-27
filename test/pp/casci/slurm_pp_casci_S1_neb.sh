#!/bin/bash -i
#
#SBATCH --job-name=pp_S1_neb
#SBATCH --output=slurme_pp_S1_neb_17.out
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
export PYSCF_EXT_PATH=$REPO_ROOT:$PYSCF_EXT_PATH
export OMP_NUM_THREADS=40

python3 pp_casci_S1_neb_17.py > pp_casci_S1_neb_17.out
