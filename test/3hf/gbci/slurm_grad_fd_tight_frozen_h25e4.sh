#!/bin/bash -i
#
#SBATCH --job-name=3hffrzh25e4
#SBATCH --output=slurme_3hf_fd_tight_frozen_h25e4_%a.out
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Tight FASSCF, cache off, grouping frozen -- the settings that took the
# finite-difference floor from 1.4e-06 to 1.0e-07. h = 2.5e-4 extends the step
# sweep under those settings; the old sweep's optimum at 1e-3 belonged to a
# noisier reference, and 2.2.5e-4 already beats it here, so the new optimum is
# expected further in.
python3 -u 01_gbci_s1_grad_check_631g.py --mode fd --step-bohr 2.5e-4 \
    --freeze-groups --no-gbci-cache --fasscf-conv-tol-grad 1e-11 \
    --reference-npz outputs/01_reference_cross_tight_tight.npz \
    --partial-dir outputs/01_partials_tight_frozen_h25e4 \
    --task-id $SLURM_ARRAY_TASK_ID --num-tasks 15 \
    > 01_grad_fd_tight_frozen_h25e4_task${SLURM_ARRAY_TASK_ID}.out
