#!/bin/bash
#SBATCH --job-name=myTest1
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --partition=g2-standard-12
#SBATCH --time=04:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

#SBATCH --output=/scratch/tgd8275/wildfire/outputs/slurm_UNETL10K_%j.out
#SBATCH --error=/scratch/tgd8275/wildfire/outputs/slurm_%j.err

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
    python eval.py \
      --model UNET_L \
      --data-dir ./data/ndws_modded \
      --checkpoint-dir ./checkpoints/unet_L_12k/UNET_L-tversky/step_0400/ckpt \
      --batch-size 32
  '
