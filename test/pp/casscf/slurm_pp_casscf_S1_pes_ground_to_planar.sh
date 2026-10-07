#!/bin/bash -i
#
#SBATCH --job-name=pp_casscf_S1_pes_g2p
#SBATCH --output=slurme_pp_casscf_S1_pes_ground_to_planar.out
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

python3 pp_casscf_S1_neb_images.py \
  --input-dir ../pes/pp_s1_pes_image_outputs/casci_ground_to_planar \
  --pattern 'pp_s1_pes_casci_ground_to_planar_*.xyz' \
  --output-dir pp_casscf_S1_pes_ground_to_planar_outputs \
  --output-prefix pp_casscf_S1_pes_ground_to_planar \
  > pp_casscf_S1_pes_ground_to_planar.out
