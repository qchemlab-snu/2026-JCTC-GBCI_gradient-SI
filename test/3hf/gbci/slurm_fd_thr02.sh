#!/bin/bash -i
#
#SBATCH --job-name=3hffd02
#SBATCH --output=logs_thr02/node_%a.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

# Every finite-difference point is single-threaded: the reference and the
# displaced energies must be computed the same way, and a threaded BLAS makes
# the last digits depend on the thread count, which 1/(2h) amplifies straight
# into the gradient.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# The grant is 10 nodes, not 10 tasks: one node per array task, filled with as
# many single-threaded points as its memory allows.  The partition mixes 503 GB
# and 64 GB nodes, so the width is read off the node at run time rather than
# fixed here -- 6 bath groups need about 2 GB a point, and a fixed 40 would
# overcommit the small nodes while a fixed 20 would waste half of the big ones.
NODES=10
GB_PER_POINT=3.0
FREE_GB=$(free -g | awk 'NR==2{print $7}')
WIDTH=$(awk -v f="$FREE_GB" -v p="$GB_PER_POINT" \
            'BEGIN{w=int(f/p); if (w>40) w=40; if (w<4) w=4; print w}')

CHUNK=logs_thr02/chunk_${SLURM_ARRAY_TASK_ID}.sh
awk -v n=$NODES -v k=$SLURM_ARRAY_TASK_ID 'NR % n == k' jobs_thr02.txt > $CHUNK

echo "node $SLURM_ARRAY_TASK_ID on $(hostname): ${FREE_GB} GB free, "\
     "$(wc -l < $CHUNK) points, $WIDTH at a time"
xargs -a $CHUNK -d '\n' -P $WIDTH -I CMD bash -c CMD
echo "node $SLURM_ARRAY_TASK_ID done at $(date)"
