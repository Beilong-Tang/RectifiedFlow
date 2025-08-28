# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training Rectified Flow on CIFAR-10 with DDPM"""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.snapshot_freq = 5000
  training.reduce_mean = True
  training.n_iters = 800000

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'rk45' ### rk45 or euler
  sampling.ode_tol = 1e-5

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm-pytorch'
  model.ema_rate = 0.999999
  model.T = 1000
  model.ch = 128
  model.ch_mult = [1,2,2,2]
  model.attn = [1]
  model.num_res_blocks = 2
  model.dropout = 0.1

  return config
