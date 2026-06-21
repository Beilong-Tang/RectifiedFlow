#!/bin/bash
#SBATCH -J train_fm
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -o _logs_slurm/train/%x_%j.out
#SBATCH -e _logs_slurm/train/%x_%j.err
source ~/miniforge3/etc/profile.d/conda.sh

conda activate sbgm

python ./main.py --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --eval_folder eval --mode train --workdir ./logs/1_rectified_flow
