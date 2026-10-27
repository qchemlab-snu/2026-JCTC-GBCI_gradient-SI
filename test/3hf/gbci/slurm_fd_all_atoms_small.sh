#!/bin/bash -i
#
#SBATCH --job-name=3hfallS
#SBATCH --output=logs_all/node_%a_small.out
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Chunks 7-9 of the same 10-way split.  The main job asks for 200 GB, which no
# longer matches any free node: the big-memory ones are taken and the fourteen
# idle ones have 64 GB.  These run there instead, 20 points at a time rather
# than 40, so the peak (about 1.3 GB per point) stays near 40 % of the node.
NODES=10
OFFSET=7
K=$((SLURM_ARRAY_TASK_ID + OFFSET))
CHUNK=logs_all/chunk_${K}.sh
awk -v n=$NODES -v k=$K 'NR % n == k' jobs_all_atoms.txt > $CHUNK

echo "chunk $K on $(hostname) ($(free -g | awk 'NR==2{print $2}') GB): $(wc -l < $CHUNK) points, 20 at a time"
xargs -a $CHUNK -d '\n' -P 20 -I CMD bash -c CMD
echo "chunk $K done at $(date)"
