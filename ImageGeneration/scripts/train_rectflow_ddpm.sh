#!/bin/bash

python recipes/train_ddp.py --workdir logs/1_rectified_flow_ddpm \
     --gpus 0,1,2,3 \
     --config configs/rectified_flow/cifar10_rf_gaussian_ddpm.py