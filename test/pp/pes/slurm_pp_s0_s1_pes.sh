#!/bin/bash -i
#SBATCH --job-name=pp_s0_s1_pes
#SBATCH --output=slurme_pp_s0_s1_pes.out
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 40core_partition

set -e

source ~/.bashrc
conda activate pyscf310
echo $CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/mygit/grad/pysfnocigrad:$PYSCF_EXT_PATH
export OMP_NUM_THREADS=40

python3 pp_s0_s1_pes_energies_to_excel.py > pp_s0_s1_pes_energies_to_excel.out
python3 pp_s1_pes_plot_figure.py > pp_s0_s1_pes_plot_figure.out
