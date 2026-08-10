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
import gc
import os
import os.path as op
import time
import sys
import json
sys.path.append(os.getcwd())

import tensorflow_gan as tfgan
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
import evaluation
import glob

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config"])

from blpytorch.utils import setup_logger

def evaluate(argv):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  config, workdir, eval_folder = FLAGS.config, FLAGS.workdir, FLAGS.eval_folder

  eval_dir = os.path.join(workdir, eval_folder)
  if not op.exists("assets/stats"):
    raise ValueError("assets/stats is not created, make sure to put stat.npz into it!")
  if not op.exists(eval_dir):
    raise ValueError(f"eval_dir {eval_dir} does not exit, abort...")
  logger = setup_logger(op.join(eval_dir, "eval_logs"), 0)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logger.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
    if not op.exists(this_sample_dir):
      logger.info(f"not existing sample dir {this_sample_dir}, continuing")
      continue
    all_logits = []
    all_pools = []

    for npz_file in tqdm.tqdm(glob.glob(op.join(this_sample_dir, "*.npz")), desc="extracting latents"):
      samples = np.load(npz_file)['samples'] # [N, H, W, C]
      latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
      gc.collect()
      if not inceptionv3:
        all_logits.append(latents["logits"])
      all_pools.append(latents['pool_3'])

    if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
    logger.info(f"Calculating IS ")
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1

    logger.info(f"Calculating FID ")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)

    logger.info(f"ckpt-{ckpt}- inception: {inception_score}, fid: {fid}")
    with open(op.join(eval_dir, f"ckpt_{ckpt}_score.json"), "w") as f:
      json.dump({"isc": inception_score.numpy().item(), "fid": fid.numpy().item()}, f)
    

  logger.info("Done...")

if __name__ == "__main__":
  app.run(evaluate)