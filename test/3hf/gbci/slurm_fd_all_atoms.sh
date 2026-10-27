#!/bin/bash -i
#
#SBATCH --job-name=3hfall
#SBATCH --output=logs_all/node_%a.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

# Every finite-difference point is single-threaded -- the reference and the
# displaced energies must be computed the same way, and a threaded BLAS makes
# the last digits of the energy depend on the thread count, which 1/(2h)
# amplifies straight into the gradient.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# The grant is 10 nodes, not 10 tasks: take one node per array task and fill it
# with 40 single-threaded points, instead of letting 10 single-core tasks idle
# 390 of the 400 cores.  --mem=200G keeps this off the 64 GB nodes, where 40
# concurrent points at ~1.3 GB would be tight.
NODES=10
awk -v n=$NODES -v k=$SLURM_ARRAY_TASK_ID 'NR % n == k' jobs_all_atoms.txt \
    > logs_all/chunk_${SLURM_ARRAY_TASK_ID}.sh

echo "node $SLURM_ARRAY_TASK_ID on $(hostname): $(wc -l < logs_all/chunk_${SLURM_ARRAY_TASK_ID}.sh) points"
xargs -a logs_all/chunk_${SLURM_ARRAY_TASK_ID}.sh -d '\n' -P 40 -I CMD bash -c CMD
echo "node $SLURM_ARRAY_TASK_ID done at $(date)"
