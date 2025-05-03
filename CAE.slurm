#!/bin/bash
#SBATCH --job-name=myTest
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=/scratch/tgd8275/wildfire/outputs/slurm_UNETL12K_%j.out
#SBATCH --error =/scratch/tgd8275/wildfire/outputs/slurm_%j.err

# Make sure Slurm can cd into a real folder:
#SBATCH --chdir=/scratch/tgd8275/wildfire/cvse-wildfire

#SBATCH --requeue

module load singularity

# Guarantee the log‐dir exists on the host before anything runs:
mkdir -p /scratch/tgd8275/wildfire/outputs

# One unbroken --overlay path, one continuous bash -c string:
singularity exec \
  --bind /scratch \
  --nv \
  --overlay /scratch/tgd8275/overlay-25GB-500K.ext3:rw \
  /scratch/tgd8275/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate cvse
    python train.py \
      --model UNET_L \
      --data-dir ./data/ndws_modded \
      --num-steps 12000 \
      --batch-size 16 \
      --loss weighted_BCE \
      --lr 2e-6 \
      --checkpoint-dir ./checkpoints/unet_L_10k
  '
