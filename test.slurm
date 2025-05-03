#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --partition=g2-standard-12
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=tgd8275@nyu.edu
#SBATCH --output=/home/tgd8275/wildfire/outputs/slurm_%j.out
#SBATCH --error=/home/tgd8275/wildfire/outputs/slurm_%j.err
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/tgd8275/overlay-
25GB-500K.ext3:rw /scratch/tgd8275/cuda11.8.86-cudnn8.7-devel-
ubuntu22.04.2.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
cd /scratch/tgd8275/wildfire/cvse-wildfire
conda activate newEnv
python temp_file.py