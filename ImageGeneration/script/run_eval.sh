#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python script/run_eval.py \
    --workdir ./logs/1_rectified_flow \
    --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
    --config.eval.num_samples 50000 --config.eval.begin_ckpt 8 --config.eval.end_ckpt 8