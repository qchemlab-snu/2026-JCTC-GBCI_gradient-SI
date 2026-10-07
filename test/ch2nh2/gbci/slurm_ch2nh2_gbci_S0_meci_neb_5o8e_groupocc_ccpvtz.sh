#!/bin/bash -i
#
#SBATCH --job-name=ch2nh2_gbci_neb_5o8e
#SBATCH --output=slurme_ch2nh2_gbci_S0_meci_neb_5o8e_groupocc_ccpvtz.out
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
echo $CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# pyscf-forge first: its pyscf.grad.gbci shadows the repo copy.
export PYSCF_EXT_PATH=/home/jiseong0626/mygit/grad/pysfnocigrad:$PYSCF_EXT_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH

export OMP_NUM_THREADS=40
python3 ch2nh2_gbci_S0_meci_neb_5o8e_groupocc_ccpvtz.py > ch2nh2_gbci_S0_meci_neb_5o8e_groupocc_ccpvtz.out
