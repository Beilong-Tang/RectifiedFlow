import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device, is_dist=False):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    print(f"resuming ckpt from {ckpt_dir}")
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    if is_dist:
      state['model'].module.load_state_dict(loaded_state['model'])
    else:
      state['model'].load_state_dict(loaded_state['model'])
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state, is_dist = False):

  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }

  if is_dist:
    saved_state['model'] = state['model'].module.state_dict()
  else:
    saved_state['model'] = state['model'].state_dict()
  torch.save(saved_state, ckpt_dir)




