import torch
from torch import Tensor, nn
from dataclasses import dataclass
from models.modules.dit_modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

@dataclass
class TrajParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

class TrajDiT(nn.Module):

    def __init__(self, params: TrajParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f'Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}')
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f'Got {params.axes_dim} but expected positional dim {pe_dim}')
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.traj_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias) for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio) for _ in range(params.depth_single_blocks)])
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, traj: Tensor, traj_ids: Tensor, cond: Tensor, cond_ids: Tensor, timesteps: Tensor, guidance: Tensor | None=None) -> Tensor:
        if traj.ndim != 3 or cond.ndim != 3:
            raise ValueError('Input traj and cond tensors must have 3 dimensions.')
        traj = self.traj_in(traj)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        cond = self.cond_in(cond)
        ids = torch.cat((cond_ids, traj_ids), dim=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            traj, cond = block(traj, cond=cond, vec=vec, pe=pe)
        traj = torch.cat((cond, traj), 1)
        for block in self.single_blocks:
            traj = block(traj, vec=vec, pe=pe)
        traj = traj[:, cond.shape[1]:, ...]
        traj = self.final_layer(traj, vec)
        return traj

    def sample(self, traj, traj_ids, cond, cond_ids, timesteps, deterministic=True, num_samples=1):
        traj = traj.clone()
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((traj.shape[0],), t_curr, dtype=traj.dtype, device=traj.device)
            pred = self(traj=traj, traj_ids=traj_ids, cond=cond, cond_ids=cond_ids, timesteps=t_vec)
            traj = traj + (t_prev - t_curr) * pred
        return traj
