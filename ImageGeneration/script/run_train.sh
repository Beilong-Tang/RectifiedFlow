#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 script/run_train.py \
    --datadir /home/btang5/work/2025/pytorch-ddpm/data \
    --workdir ./logs/1_rectified_flow_train \
    --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
    --config.training.batch_size 32