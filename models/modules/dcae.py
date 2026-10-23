import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, OmegaConf
from safetensors.torch import load_file as load_sft
from einops import rearrange
from typing import Optional, Tuple

from models.modules.dcae_layers.act import build_act
from models.modules.dcae_layers.norm import build_norm
from models.modules.dcae_layers.ops import ChannelDuplicatingPixelUnshuffleUpSampleLayer, ConvLayer, ConvPixelShuffleUpSampleLayer, ConvPixelUnshuffleDownSampleLayer, EfficientViTBlock, IdentityLayer, OpSequential, PixelUnshuffleChannelAveragingDownSampleLayer, ResBlock, ResidualBlock
from utils.utils import format_number

__all__ = ["DCAE", "dc_ae_f32c32"]

@dataclass
class EncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False

@dataclass
class DecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"

@dataclass
class DCAEConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    use_quant_conv: bool = False

    add_encoder_temporal: bool = False
    add_decoder_temporal: bool = False
    condition_frames: int = 1
    token_size: int = 32 * 64

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"

def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block

def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage

def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "Conv":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act=None,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block

def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block

def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif factor == 2:
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block

def build_encoder_project_out_block(
    in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], shortcut: Optional[str]
):
    block = OpSequential(
        [
            build_norm(norm),
            build_act(act),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            ),
        ]
    )
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block

def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block

def build_decoder_project_out_block(
    in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    elif factor == 2:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:

    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def nonlinearity(x):

    return x*torch.sigmoid(x)

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)

def zero_initialize(module):
    for param in module.parameters():
        nn.init.zeros_(param)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, dim, multiple_of=256, ffn_dim_multiplier=None, norm_eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(n_heads=n_heads, dim=dim)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
    ):

        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Attention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.n_kv_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )
        self.n_rep = 1

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        keys = repeat_kv(
            keys, self.n_rep
        )
        values = repeat_kv(
            values, self.n_rep
        )

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(
            1, 2
        )
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )

        self.project_in = build_encoder_project_in_block(
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            stage = build_stage_main(
                width=width, depth=depth, block_type=block_type, norm=cfg.norm, act=cfg.act, input_width=width
            )

            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=cfg.downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width,
                    shortcut=cfg.downsample_shortcut,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            in_channels=cfg.width_list[-1],
            out_channels=2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
            shortcut=cfg.out_shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
            shortcut=cfg.in_shortcut,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=cfg.upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=cfg.upsample_shortcut,
                )
                stage.append(upsample_block)

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x

class DCAE(nn.Module):
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)

        self.add_encoder_temporal = cfg.add_encoder_temporal
        self.add_decoder_temporal = cfg.add_decoder_temporal
        self.temporal_block_num = 5

        if self.add_encoder_temporal or self.add_decoder_temporal:
            self.condition_frames = cfg.condition_frames
            self.token_size = cfg.token_size
            self.causal_time_block = nn.Sequential(*[TransformerBlock(n_heads=8, dim=32) for _ in range(self.temporal_block_num)])
            self.space_block = nn.Sequential(*[TransformerBlock(n_heads=8, dim=32) for _ in range(self.temporal_block_num)])
            zero_initialize(self.causal_time_block)
            zero_initialize(self.space_block)

            print("DCAE Params Total:", format_number(sum(p.numel() for p in self.parameters())))
            print("Encoder:", format_number(sum(p.numel() for p in self.encoder.parameters())))
            print("Decoder:", format_number(sum(p.numel() for p in self.decoder.parameters())))
            print("Causal Time Block:", format_number(sum(p.numel() for p in self.causal_time_block.parameters())))
            print("Space Block:", format_number(sum(p.numel() for p in self.space_block.parameters())))

            self.freqs_cis_time_vid = precompute_freqs_cis(
                4,
                self.condition_frames,
                1000,
            ).cuda()
            self.freqs_cis_time_img = precompute_freqs_cis(
                4,
                1,
                1000,
            ).cuda()
            self.freqs_cis_space_vid = precompute_freqs_cis(
                4,
                self.token_size,
                1000,
            ).cuda()
            self.freqs_cis_space_img = precompute_freqs_cis(
                4,
                self.token_size,
                1000,
            ).cuda()

        if self.cfg.pretrained_path is not None:
            self.load_model()

    def load_model(self):
        if self.cfg.pretrained_source == "dc-ae":
            if self.cfg.pretrained_path.endswith(".safetensors"):
                state_dict = load_sft(self.cfg.pretrained_path, device="cpu")
            else:
                state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["model"]
            self.load_state_dict(state_dict)
            print(f"load from {self.cfg.pretrained_path}")
            del state_dict
        else:
            raise NotImplementedError

    def spatial_temporal_blocks(self, x):
        _, f, l, _ = x.shape
        matrix = torch.tril(torch.ones(f, f))
        time_causal_mask = torch.where(matrix==0, float('-inf'), matrix)
        time_causal_mask = torch.where(matrix==1, 0, time_causal_mask)
        mask_time = time_causal_mask.contiguous().cuda()

        x_b = x.clone()
        xx = rearrange(x, 'b f l c -> (b f) l c')
        for i in range(self.temporal_block_num):
            xx = rearrange(xx, '(b f) l c -> (b l) f c', l=l, f=f)
            xx = self.causal_time_block[i](xx, self.freqs_cis_time_img if f == 1 else self.freqs_cis_time_vid, mask_time)
            xx = rearrange(xx, '(b l) f c -> (b f) l c', l=l, f=f)
            xx = self.space_block[i](xx, self.freqs_cis_space_img if f == 1 else self.freqs_cis_space_vid)
        x = rearrange(xx, '(b f) l c -> b f l c', f=f)
        return x

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.decoder.num_stages - 1)

    def encode(self, x: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if self.add_encoder_temporal:
            _, _, h, w = x.shape
            x = rearrange(x, "(b f) c h w -> b f (h w) c", f=self.condition_frames if is_video else 1)
            x = self.spatial_temporal_blocks(x)
            x = rearrange(x, "b f (h w) c -> (b f) c h w", h=h, w=w)
        return x

    def decode(self, x: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        if self.add_decoder_temporal:
            _, _, h, w = x.shape
            x = rearrange(x, "(b f) c h w -> b f (h w) c", f=self.condition_frames if is_video else 1)
            x = self.spatial_temporal_blocks(x)
            x = rearrange(x, "b f (h w) c -> (b f) c h w", h=h, w=w)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if self.add_encoder_temporal:
            _, _, h, w = x.shape
            x = rearrange(x, "(b f) c h w -> b f (h w) c", f=self.condition_frames if is_video else 1)
            x = self.spatial_temporal_blocks(x)
            x = rearrange(x, "b f (h w) c -> (b f) c h w", h=h, w=w)
        if self.add_decoder_temporal:
            _, _, h, w = x.shape
            x = rearrange(x, "(b f) c h w -> b f (h w) c", f=self.condition_frames if is_video else 1)
            x = self.spatial_temporal_blocks(x)
            x = rearrange(x, "b f (h w) c -> (b f) c h w", h=h, w=w)
        x = self.decoder(x)
        return x

def dc_ae_f32c32(name: str,
                 pretrained_path: Optional[str] = None,
                 add_encoder_temporal: bool = False,
                 add_decoder_temporal: bool = False,
                 condition_frames: int = 7,
                 token_size: int = 32 * 64) -> DCAEConfig:
    if name in ["dc-ae-f32c32-in-1.0", "dc-ae-f32c32-mix-1.0"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[0,4,8,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu]"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    cfg.add_encoder_temporal = add_encoder_temporal
    cfg.add_decoder_temporal = add_decoder_temporal
    cfg.condition_frames = condition_frames
    cfg.token_size = token_size
    return cfg
