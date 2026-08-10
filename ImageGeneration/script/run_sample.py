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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """
import io
import os
import os.path as op
import time
import sys
sys.path.append(os.getcwd())

import numpy as np
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import sde_lib
from absl import flags
import torch
from utils import restore_checkpoint
from absl import app
from ml_collections.config_flags import config_flags
import tqdm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("datadir", None, "datadir")
flags.mark_flags_as_required(["workdir", "config", "datadir"])

from blpytorch.utils import setup_all_ddp

def sample(argv):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  config, workdir, eval_folder, datadir = FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.datadir

  eval_dir = os.path.join(workdir, eval_folder)
  os.makedirs(eval_dir, exist_ok=True)

  is_ddp, rank, logger, world_size, device = setup_all_ddp(None, eval_dir, init_process=False)
  logger.info(f"rank {rank} of world size {world_size} is inited", all=True)
  config.device = device

  # Build data pipeline
  train_ds, eval_ds = datasets.get_dataset(config,
                                              datadir,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  begin_ckpt = config.eval.begin_ckpt
  logger.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not op.exists(ckpt_filename):
      if not waiting_message_printed:
        logger.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    # Generate samples and compute IS/FID/KID when enabled
    num_samples = config.eval.num_samples // world_size
    assert num_samples % world_size == 0
    if config.eval.enable_sampling:
      num_sampling_rounds = num_samples // config.eval.batch_size + 1
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      os.makedirs(this_sample_dir, exist_ok=True)
      for r in tqdm.tqdm(range(num_sampling_rounds), desc=f'ckpt: {ckpt}'):
        # Directory to save samples. Different for each host to avoid writing conflicts
        if op.exists(op.join(this_sample_dir, f"samples_rank_{rank}_{r}.npz")):
          continue
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with open(
            os.path.join(this_sample_dir, f"samples_rank_{rank}_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

  logger.info(f"rank {rank} Done...")

if __name__ == "__main__":
  app.run(sample)