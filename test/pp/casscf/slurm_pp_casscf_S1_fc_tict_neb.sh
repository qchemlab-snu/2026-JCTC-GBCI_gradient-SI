#!/bin/bash -i
#
#SBATCH --job-name=pp_casscf_S1_fc_tict_neb
#SBATCH --output=slurme_pp_casscf_S1_fc_tict_neb.out
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=672:00:00
#SBATCH -p 32core_partition

source ~/.bashrc
conda activate pyscf310
echo $CONDA_PREFIX

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYSCF_EXT_PATH=/home/jiseong0626/labgit/pyscf-forge:$PYSCF_EXT_PATH
export OMP_NUM_THREADS=32

# CASSCF S0 minimum -> CASSCF S1 TICT, restarted from the best band of the
# stopped pp_casscf_S1_neb run (step 36, before image 1 collapsed into the
# fixed endpoint) instead of a fresh IDPP interpolation.
#
# The band is wanted for a PES figure, not for a stationary point, so fmax is
# loosened from 0.05 to 0.10 and the step count is capped at 60. The seed band
# already sits near 0.14, and the FC endpoint is not stationary anyway, so a
# tighter target would only buy a shape the figure cannot show.
python3 pp_casscf_S1_planar_tict_neb.py \
  --seed-traj pp_casscf_S1_neb_outputs/pp_casscf_S1_neb.traj \
  --seed-traj-step 36 \
  --initial outputs/pp_casscf_S0/pp_casscf_S0_optimized.xyz \
  --final outputs/pp_casscf_S1/pp_casscf_S1_optimized.xyz \
  --output-dir pp_casscf_S1_fc_tict_neb_outputs \
  --prefix pp_casscf_S1_fc_tict_neb \
  --path-label casscf_S0_minimum_to_casscf_S1_TICT \
  --k 30.0 \
  --fmax 0.10 \
  --max-steps 60 \
  > pp_casscf_S1_fc_tict_neb.out
