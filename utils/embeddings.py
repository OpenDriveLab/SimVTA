import torch
import math
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 100,
):

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_fourier_embeds_from_coordinates(embed_dim, coordinates, max_period: int = 100,):

    half_embed_dim = embed_dim // 2
    B, N, C = coordinates.shape
    emb = max_period ** (torch.arange(half_embed_dim, dtype=torch.float32, device=coordinates.device) / half_embed_dim)
    emb = emb[None, None, None, :] * coordinates[:, :, :, None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

if __name__ == '__main__':

    coordinates = torch.rand((1, 2, 3))
    emb = get_fourier_embeds_from_coordinates(6, coordinates)
    a, b, c = torch.split(emb, dim=2, split_size_or_sections=1)
    print(emb)
    print(emb.shape)
    print(a.shape)
