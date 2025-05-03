#!/bin/bash
#SBATCH --job-name=myTest
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --partition=g2-standard-12
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=tgd8275@nyu.edu
#SBATCH --output=/scratch/tgd8275/wildfire/outputs/slurm_%j.out
#SBATCH --error=/scratch/tgd8275/wildfire/outputs/slurm_%j.err
#SBATCH --requeue

module load singularity

singularity exec \
  --bind /scratch \
  --nv \
  --overlay /scratch/tgd8275/overlay-25GB-500K.ext3:rw \
  /scratch/tgd8275/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate cvse
    cd /scratch/tgd8275/wildfire/cvse-wildfire

    python train.py \
      --model UNET \
      --data-dir ./data/ndws_modded \
      --num-steps 2000 \
      --batch-size 64 \
      --loss weighted_BCE \
      --lr 5e-7 \
      --checkpoint-dir ./checkpoints/unet_ubce1
  "
