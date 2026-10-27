#!/bin/bash -i
#
#SBATCH --job-name=pp_casscf_S0
#SBATCH --output=slurme_pp_casscf_S0.out
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
echo "$CONDA_PREFIX"

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=40

stage="${1:-all}"
python3 pp_casscf_S0.py --stage "$stage" > "pp_casscf_S0_${stage}.out"
