#!/bin/bash
#SBATCH -J eval_fm
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 96:00:00
#SBATCH -o _logs_slurm/eval/%x_%j.out
#SBATCH -e _logs_slurm/eval/%x_%j.err
set -e

source ~/miniforge3/etc/profile.d/conda.sh

conda activate sbgm_no_tf

## Generation script

# python ./main.py --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --eval_folder eval --mode eval --workdir ./logs/1_rectified_flow --config.eval.enable_sampling  --config.eval.batch_size 1024 --config.eval.num_samples 50000 --config.eval.begin_ckpt 13

## Evaluation script using torch-fidelity
echo "[FID and IS]"
fidelity --isc --fid --input1 /work/btang1/RectifiedFlow/ImageGeneration/logs/1_rectified_flow/eval/ckpt_13/imgs --input2 cifar10-train