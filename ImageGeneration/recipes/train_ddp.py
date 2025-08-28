## Use ddp for training
import importlib
import argparse
import yaml
import os
from functools import partial
import random
import numpy as np
import copy
import os.path as op
from pathlib import Path
from tqdm import trange
from dataclasses import asdict, dataclass
from absl import app, flags
import sys
import numpy as np


import torch 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from ml_collections.config_flags import config_flags

sys.path.append(os.getcwd())

from models import utils as mutils
from models.ema import ExponentialMovingAverage
from models.ddpm import *
import losses
import sampling
from utils import save_checkpoint, restore_checkpoint
from _utils.logger import setup_logger
import sde_lib
import datasets
from models.ddpm_pytorch import *

def build_flags():
    f = flags.FlagValues()
    flags.DEFINE_integer("port", 12355, "ddp port", flag_values=f)
    flags.DEFINE_string("dist_backend", "nccl", "ddp backend", flag_values=f)
    flags.DEFINE_bool("resume", False, "resume training", flag_values=f)
    flags.DEFINE_string("gpus", "0,1,2,3", "visible GPUs", flag_values=f)
    flags.DEFINE_integer("seed", 1234, "random seed", flag_values=f)
    flags.DEFINE_string("workdir", None, "output directory", flag_values=f)
    flags.mark_flags_as_required(["workdir"], flag_values=f)
    # Config flag (this is the trick)
    config_flags.DEFINE_config_file(
        "config", None, "Training configuration.", lock_config=True, flag_values=f
    )
    return f

def gather_all(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    # Allocate list for all gathered tensors
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    # Concatenate along the first dimension
    gather_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in gather_list]
    return torch.cat(gather_list, dim=0)

def infiniteloop(dataloader, epoch=0):
    epoch = epoch
    while True:
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(int(epoch))
        for x, y in iter(dataloader):
            yield x
        epoch +=1

def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return SEED

## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    if dist.is_initialized():
        # Optional: try a barrier to flush pending work
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def main(rank, arg_list):
    f = build_flags()
    f(arg_list)
    print(f"INFO: [rank[{rank}] | {len(f.gpus.split(','))}] inited...")
    setup(rank, len(f.gpus.split(',')), f.dist_backend, f.port)
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    setup_seed(f.seed, rank)

    workdir = f.workdir
    config = f.config
    if op.exists(op.join(workdir, "checkpoints-meta", "checkpoint.pth")):
        if not f.resume:
            raise Exception(f"workdir {workdir} exists, aborting...")
    
    sample_dir=op.join(workdir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    if rank == 0:
        writer = SummaryWriter(tb_dir)
    
    score_model = mutils.create_model(config, data_parallel=False) # Return raw model file
    # DDP
    score_model = DDP(score_model, device_ids=[rank])
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Load CIFAR10 dataset
    dataset = CIFAR10(
        root='/home/btang5/work/2025/data/cifar10', train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1,1]
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size // len(f.gpus.split(',')), 
        shuffle=False,
        num_workers=config.data.num_workers,  
        worker_init_fn = seed_worker,
        sampler=DistributedSampler(dataset),
        pin_memory=True)
    
    # Eval dataset
    eval_dataset = CIFAR10(
        root='/home/btang5/work/2025/data/cifar10', train=False, download=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1,1]
        ]))
    eval_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size // len(f.gpus.split(',')), 
        shuffle=False,
        num_workers=config.data.num_workers,  
        worker_init_fn = seed_worker,
        sampler=DistributedSampler(dataset, shuffle=False),
        pin_memory=True)
    for _x, _ in eval_dataloader:
        eval_batch = _x
        break
    # Logging
    logger = setup_logger(op.join(f.workdir, "logging"), rank)
    logger.info(f"len tr_dataloder dataset for rank {rank}: {len(dataloader) * dataloader.batch_size}")
    dataloader = infiniteloop(dataloader, epoch = (initial_step * config.training.batch_size) / len(dataset)+ 1)

    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
    sampling_eps = 1e-3

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting)
    # Building sampling functions
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    
    num_train_steps = config.training.n_iters
    logger.info("Starting training loop at step %d." % (initial_step))

    if rank == 0:
        pbar = trange(initial_step, num_train_steps + 1)
    else:
        pbar = range(initial_step, num_train_steps + 1)

    dist.barrier()
    # Training loop
    for step in pbar:
        batch = next(dataloader) # [-1,1]
        batch = batch.cuda()
        loss = train_step_fn(state, batch)

        if rank == 0 and step % config.training.log_freq == 0:
            # logger.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss.item(), step)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        # Save a temporary checkpoint to resume training after pre-emption periodically
        if rank == 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
        
        if step % config.training.eval_freq == 0:
            eval_batch = eval_batch.cuda()
            eval_loss = eval_step_fn(state, eval_batch)
            eval_loss_all = gather_all(eval_loss).mean()
            if rank == 0:
                # logger.info("step: %d, eval_loss: %.5e" % (step, eval_loss_all.item()))
                writer.add_scalar("eval_loss", eval_loss_all.item(), step)
        
        # Save a checkpoint periodically and generate samples if needed
        if step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            if rank == 0:
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state, ddp = True)

                if config.training.snapshot_sampling:
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample, n = sampling_fn(score_model)
                    ema.restore(score_model.parameters())
                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    os.makedirs(this_sample_dir)
                    nrow = int(np.sqrt(sample.shape[0]))
                    image_grid = make_grid(sample, nrow, padding=2)
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    np.save(os.path.join(this_sample_dir, "sample.npy"), sample)
                    save_image(image_grid, os.path.join(this_sample_dir, "sample.png"))
            dist.barrier()

    cleanup()


if __name__ == "__main__":
    args = build_flags()
    arg_list = sys.argv
    args(arg_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if len(args.gpus.split(",")) > 1:
        print("running DDP")
        mp.spawn(main, args=(arg_list,), nprocs=len(args.gpus.split(",")), join=True)
        print("Done")
    else:
        print("Not running DDP")
        main(0, arg_list)
    pass