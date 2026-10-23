import math
from typing import Callable
import torch
from einops import repeat
from torch import Tensor

def prepare_ids(bs, h, w, seq_len, traj_len, device='cuda') -> dict[str, Tensor]:
    img_ids = torch.zeros(h, w, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w)[None, :]
    img_ids = repeat(img_ids, 'h w c -> b (h w) c', b=bs).to(device)
    cond_ids = torch.zeros(bs, seq_len, 3).to(device)
    traj_ids = torch.zeros(bs, traj_len, 3)
    traj_ids[..., -1] = traj_ids[..., -1] + torch.arange(traj_len)[None, :]
    traj_ids = traj_ids.to(device)
    return (img_ids, cond_ids, traj_ids)

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1: float=256, y1: float=0.5, x2: float=4096, y2: float=1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(num_steps: int, image_seq_len: int, base_shift: float=0.5, max_shift: float=1.15, shift: bool=True) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    timesteps = 1 - timesteps
    return timesteps.tolist()
