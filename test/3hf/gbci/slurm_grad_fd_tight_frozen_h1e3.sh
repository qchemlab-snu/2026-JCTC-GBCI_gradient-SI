#!/bin/bash -i
#
#SBATCH --job-name=3hffrz1e3
#SBATCH --output=slurme_3hf_fd_tight_frozen_h1e3_%a.out
#SBATCH --array=0-14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH -p 40core_partition

source ~/.bashrc
conda activate pyscf310
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=${PYSCF_EXT_PATH:-$HOME/labgit/pyscf-forge}

# One thread per task, as every finite-difference reference here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# h = 1e-3 again, this time with the grouping frozen to the reference. The
# unfrozen run lost C1 x and C2 x to the membership guard, which are the two
# components that used to carry the largest discrepancy; freezing makes the
# displaced points measure the surface the analytic gradient differentiates,
# so all fifteen should come back.
#
# --freeze-groups is the default now and is written out only to keep the
# command self-describing.
python3 -u 01_gbci_s1_grad_check_631g.py --mode fd --step-bohr 1e-3 \
    --freeze-groups --no-gbci-cache --fasscf-conv-tol-grad 1e-11 \
    --reference-npz outputs/01_reference_cross_tight_tight.npz \
    --partial-dir outputs/01_partials_tight_frozen_h1e3 \
    --task-id $SLURM_ARRAY_TASK_ID --num-tasks 15 \
    > 01_grad_fd_tight_frozen_h1e3_task${SLURM_ARRAY_TASK_ID}.out
