#!/bin/bash -i
#
#SBATCH --job-name=gbci_so2_grad_fd_sweep
#SBATCH --output=slurme_so2_grad_fd_sweep.out
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

export OMP_NUM_THREADS=40
# python3 fe2s2_re.py > fe2s2_casci_re.out
#python3 ch2nh2_casci_s0.py > ch2nh2_S0_4o6e_casci.out
python3 gbci_so2_grad_fd_sweep.py > gbci_so2_grad_fd_sweep.out
#python3 gbci_so2_S1S2_meci_step2.py > gbci_so2_S1S2_meci_step2.out
