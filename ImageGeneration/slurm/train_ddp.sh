#!/bin/bash
#SBATCH -J train_fm_ddp
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -o _logs_slurm/train_ddp/%x_%j.out
#SBATCH -e _logs_slurm/train_ddp/%x_%j.err
source ~/miniforge3/etc/profile.d/conda.sh

conda activate sbgm

python ./main.py --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp_ddp.py --eval_folder eval --mode train --workdir ./logs/1_rectified_flow_ddp --num_gpus 2
