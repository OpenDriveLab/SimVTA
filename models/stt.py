import random
import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from utils.rope_2d import compute_axial_cis, precompute_1d_freqs_cis, apply_rotary_emb, apply_1d_rotary_emb
from utils.embeddings import get_fourier_embeds_from_coordinates

class GPTConfig:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.pose_tokens_num = config.token_size_dict['pose_tokens_size']
        self.img_tokens_num = config.token_size_dict['img_tokens_size']
        self.yaw_token_size = config.token_size_dict['yaw_token_size']
        self.total_tokens_num = config.token_size_dict['total_tokens_size']

        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num
        self.freqs_cis_singlescale = compute_axial_cis(dim = config.n_embd  // self.n_head, end_x = self.patch_size[0], end_y = self.patch_size[1], theta = 1000.0)

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if T > self.pose_tokens_num+self.yaw_token_size:
            q_B_scale2_d = q[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            q_out, k_out = apply_rotary_emb(q_B_scale2_d, k_B_scale2_d, freqs_cis=self.freqs_cis_singlescale[:T-self.pose_tokens_num-self.yaw_token_size])
            q = torch.cat([q[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], q_out], dim=2)
            k = torch.cat([k[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], k_out], dim=2)

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        ).transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y

class CausalSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x

class SpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.pose_tokens_num = config.token_size_dict['pose_tokens_size']
        self.img_tokens_num = config.token_size_dict['img_tokens_size']
        self.yaw_token_size = config.token_size_dict['yaw_token_size']
        self.total_tokens_num = config.token_size_dict['total_tokens_size']

        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num
        self.freqs_cis_singlescale = compute_axial_cis(dim = config.n_embd  // self.n_head, end_x = self.patch_size[0], end_y = self.patch_size[1], theta = 1000.0)

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if T > self.pose_tokens_num+self.yaw_token_size:
            q_B_scale2_d = q[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            q_out, k_out = apply_rotary_emb(q_B_scale2_d, k_B_scale2_d, freqs_cis=self.freqs_cis_singlescale[:T-self.pose_tokens_num-self.yaw_token_size])
            q = torch.cat([q[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], q_out], dim=2)
            k = torch.cat([k[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], k_out], dim=2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        ).transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y

class SpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x

class CausalTimeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask.to(q.dtype),
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        ).transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y

class CausalTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalTimeSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x

class CausalTimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_time_block = CausalTimeBlock(config)
        self.space_block = SpaceBlock(config)

    def forward(self, x, attn_mask):

        b, f, l, c = x.shape
        x = rearrange(x, 'b f l c -> (b l) f c')
        x = self.causal_time_block(x, attn_mask)
        x = rearrange(x, '(b l) f c -> (b f) l c', b=b, l=l, f=f)
        x = self.space_block(x)
        x = rearrange(x, '(b f) l c -> b f l c', b=b, f=f)
        return x

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, block_size, n_layer=[12, 6], n_head=8, n_embd=1024,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 local_rank=0, condition_frames = 3, latent_size = (32, 32),
                 token_size_dict=None, vae_emb_dim = 8, temporal_block=1,
                 pose_x_vocab_size=512, pose_y_vocab_size=512, yaw_vocab_size=512,
                 ):
        super().__init__()
        config = GPTConfig(block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked,
                           patch_size=latent_size,
                           condition_frames=condition_frames,
                           token_size_dict=token_size_dict)

        self.C = n_embd
        self.Cvae = vae_emb_dim
        self.yaw_pose_emb_dim = 512
        self.pose_x_vocab_num = pose_x_vocab_size
        self.pose_y_vocab_num = pose_y_vocab_size
        self.yaw_vocab_num = yaw_vocab_size
        self.latent_size = latent_size
        self.temporal_block = temporal_block
        self.img_projector = nn.Sequential(
            nn.Linear(self.Cvae, self.C // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.C//2, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.pose_x_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.pose_y_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.yaw_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.causal_time_space_num = config.n_layer[0]
        self.auto_regressive_num = config.n_layer[1]

        self.local_rank = local_rank
        self.img_token_size = token_size_dict['img_tokens_size']
        self.total_token_size = token_size_dict['total_tokens_size']
        self.yaw_token_size = token_size_dict['yaw_token_size']
        self.pose_token_size = token_size_dict['pose_tokens_size']
        self.prefix_size = self.total_token_size - self.img_token_size
        self.condition_frames = condition_frames

        self.time_emb = nn.Parameter(torch.zeros(50, self.C))
        nn.init.normal(self.time_emb.data, mean=0, std=0.02)
        self.begin_ends = []

        self.causal_time_space_blocks = nn.Sequential(*[CausalTimeSpaceBlock(config) for _ in range(self.causal_time_space_num)])

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

        matrix = torch.tril(torch.ones(condition_frames, condition_frames))
        time_causal_mask = torch.where(matrix==0, float('-inf'), matrix)
        time_causal_mask = torch.where(matrix==1, 0, time_causal_mask)
        self.mask_time = time_causal_mask.contiguous().cuda()

        matrix_1 = torch.ones(self.total_token_size, self.total_token_size)
        for i in range(0, self.prefix_size):
            matrix_1[i, self.prefix_size:] = 0
        seq_causal_mask = torch.where(matrix_1==0, float('-inf'), matrix_1)
        seq_causal_mask = torch.where(matrix_1==1, 0, seq_causal_mask)
        beta = 0.1
        space_weight = torch.zeros(self.total_token_size, self.total_token_size)
        space_weight[:, 0] = 2
        space_weight[:, 1] = 1
        seq_causal_mask = seq_causal_mask + space_weight * beta
        self.mask_space = seq_causal_mask.contiguous().cuda()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_yaw_pose_emb(self, pose_indices, yaw_indices):
        if pose_indices == None:
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            yaw_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                yaw_indices_normalize)
            return yaw_emb, None, None
        elif pose_indices is not None and (pose_indices.shape[-1]==1):
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                torch.cat([yaw_indices_normalize, pose_x_indices_normalize], dim=-1), )
            yaw_emb, pose_x_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            return yaw_emb, pose_x_emb, None
        else :
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            pose_y_indices_normalize = pose_indices[:, :, 1:2] / self.pose_y_vocab_num
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                torch.cat([yaw_indices_normalize, pose_x_indices_normalize, pose_y_indices_normalize], dim=-1),
                )
            yaw_emb, pose_x_emb, pose_y_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            return yaw_emb, pose_x_emb, pose_y_emb

    def forward(self, feature_total, pose_indices_total, yaw_indices_total, drop_feature=0):

        B, F, _, _ = feature_total.shape
        F = F - 1
        yaw_emb_total, pose_x_emb_total, pose_y_emb_total = self.get_yaw_pose_emb(pose_indices_total, yaw_indices_total)

        pose_x_token_embeddings = self.pose_x_projector(pose_x_emb_total)
        pose_y_token_embeddings = self.pose_y_projector(pose_y_emb_total)
        yaw_token_embeddings = self.yaw_projector(yaw_emb_total)
        feature_embeddings = self.img_projector(feature_total)
        pose_x_token_embeddings = rearrange(pose_x_token_embeddings, "B (F T) L C -> B F (L T) C", F=F+1, T=self.temporal_block)
        pose_y_token_embeddings = rearrange(pose_y_token_embeddings, "B (F T) L C -> B F (L T) C", F=F+1, T=self.temporal_block)
        yaw_token_embeddings = rearrange(yaw_token_embeddings, "B (F T) L C -> B F (L T) C", F=F+1, T=self.temporal_block)

        pro = random.random()
        if pro > drop_feature:

            input_pose_x_token_embeddings = pose_x_token_embeddings[:, :-1, ...]
            input_pose_y_token_embeddings = pose_y_token_embeddings[:, :-1, ...]
            input_yaw_token_embeddings = yaw_token_embeddings[:, :-1, ...]
            input_feature_embeddings = feature_embeddings[:, :-1, ...]

            yaw_pose_scale_token_embeddings = torch.cat([input_yaw_token_embeddings, input_pose_x_token_embeddings, input_pose_y_token_embeddings, input_feature_embeddings], dim=2)

            time_emb_F = self.time_emb[:F, :].unsqueeze(0)
            time_emb_F = torch.repeat_interleave(time_emb_F[:, :, None, :], self.total_token_size, dim=2)

            time_space_token_embeddings = yaw_pose_scale_token_embeddings + time_emb_F

            for i in range(self.causal_time_space_num):
                time_space_token_embeddings = self.causal_time_space_blocks[i](time_space_token_embeddings, self.mask_time)
            auto_regressive_token_embeddings = rearrange(time_space_token_embeddings, 'B F L C -> (B F) L C', B=B, F=F)

        target_pose_x_token_embeddings = pose_x_token_embeddings[:, self.temporal_block:, ...].reshape(B*F, self.temporal_block*self.C)
        target_pose_y_token_embeddings = pose_y_token_embeddings[:, self.temporal_block:, ...].reshape(B*F, self.temporal_block*self.C)
        target_yaw_token_embeddings = yaw_token_embeddings[:, self.temporal_block:, ...].reshape(B*F, self.temporal_block*self.C)
        pose_emb = torch.cat([target_pose_x_token_embeddings, target_pose_y_token_embeddings, target_yaw_token_embeddings], dim=1)

        out = {
            'logits': auto_regressive_token_embeddings,
            'pose_emb': pose_emb
        }
        return out

    @torch.no_grad()
    def evaluate(self, feature, pose_total, yaw_total, sample_last=True):

        yaw_emb_total, pose_x_emb_total, pose_y_emb_total = self.get_yaw_pose_emb(pose_total, yaw_total)

        pose_x_token_embeddings = self.pose_x_projector(pose_x_emb_total)
        pose_y_token_embeddings = self.pose_y_projector(pose_y_emb_total)
        yaw_token_embeddings = self.yaw_projector(yaw_emb_total)

        input_pose_x_token_embeddings = pose_x_token_embeddings[:, :-1, ...]
        input_pose_y_token_embeddings = pose_y_token_embeddings[:, :-1, ...]
        input_yaw_token_embeddings = yaw_token_embeddings[:, :-1, ...]

        feature_embeddings = self.img_projector(feature)
        yaw_pose_scale_token_embeddings = torch.cat([input_yaw_token_embeddings, input_pose_x_token_embeddings, input_pose_y_token_embeddings, feature_embeddings], dim=2)
        B, F, _, _ = yaw_pose_scale_token_embeddings.shape

        time_emb_F = self.time_emb[:F, :].unsqueeze(0)
        time_emb_F = torch.repeat_interleave(time_emb_F[:, :, None, :], self.total_token_size, dim=2)

        time_space_token_embeddings = yaw_pose_scale_token_embeddings + time_emb_F

        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](time_space_token_embeddings, self.mask_time)

        if sample_last:
            time_space_token_embeddings = time_space_token_embeddings[:, -1:, :, :]
            target_pose_x_token_embeddings = pose_x_token_embeddings[:, -1, ...].reshape(B, self.C)
            target_pose_y_token_embeddings = pose_y_token_embeddings[:, -1, ...].reshape(B, self.C)
            target_yaw_token_embeddings = yaw_token_embeddings[:, -1, ...].reshape(B, self.C)
        else:
            target_pose_x_token_embeddings = pose_x_token_embeddings[:, 1:, ...].reshape(B*F, self.C)
            target_pose_y_token_embeddings = pose_y_token_embeddings[:, 1:, ...].reshape(B*F, self.C)
            target_yaw_token_embeddings = yaw_token_embeddings[:, 1:, ...].reshape(B*F, self.C)

        auto_regressive_token_embeddings = rearrange(time_space_token_embeddings, 'B F L C -> (B F) L C')
        pose_emb = torch.cat([target_pose_x_token_embeddings, target_pose_y_token_embeddings, target_yaw_token_embeddings], dim=1)
        return auto_regressive_token_embeddings, pose_emb

    def get_pose_emb(self, pose, yaw):
        yaw_emb, pose_x_emb, pose_y_emb = self.get_yaw_pose_emb(pose, yaw)
        pose_x_token_embeddings = self.pose_x_projector(pose_x_emb)
        pose_y_token_embeddings = self.pose_y_projector(pose_y_emb)
        yaw_token_embeddings = self.yaw_projector(yaw_emb)
        pose_emb = torch.cat([pose_x_token_embeddings, pose_y_token_embeddings, yaw_token_embeddings], dim=-1).reshape(-1, self.C*3)
        return pose_emb

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
