from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch.utils.data.distributed import DistributedSampler


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def get_dataset(config, uniform_dequantization=False, evaluation=False, is_dist = False):
    assert uniform_dequantization is False

    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

    if config.data.dataset == 'CIFAR10':
        transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1 to 1
          ])
        train_dataset = CIFAR10(
          root='/work/btang1/pytorch-ddpm/data', train=True, download=True,
          transform=transform
        )
        if not is_dist:
          train_dataloader = torch.utils.data.DataLoader(
              train_dataset, batch_size=batch_size, shuffle=True,
              num_workers=4, drop_last=False)
        else:
           train_dataloader = torch.utils.data.DataLoader(
              train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset), pin_memory=True, persistent_workers=True,
              num_workers=4, drop_last=False)
        
        eval_dataset = CIFAR10(
          root='/work/btang1/pytorch-ddpm/data', train=False, download=True,
          transform=transform
        )
        if not is_dist:
          eval_dataloader = torch.utils.data.DataLoader(
              eval_dataset, batch_size=batch_size, shuffle=False,
              num_workers=4, drop_last=False)
        else:
           eval_dataloader = torch.utils.data.DataLoader(
              eval_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(eval_dataset), pin_memory=True, persistent_workers=True,
              num_workers=4, drop_last=False)
        return train_dataloader, eval_dataloader
    else:
        raise Exception(f"current dataset: {config.data.dataset}, is not supported")
    
