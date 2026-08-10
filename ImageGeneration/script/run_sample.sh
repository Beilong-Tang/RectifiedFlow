#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 script/run_sample.py \
    --datadir /home/btang5/work/2025/pytorch-ddpm/data \
    --workdir ./logs/1_rectified_flow \
    --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
    --config.eval.enable_sampling  --config.eval.batch_size 128 --config.eval.num_samples 50000 \
    --config.eval.begin_ckpt 8 --config.eval.end_ckpt 8