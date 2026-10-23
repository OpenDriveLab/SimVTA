import torch
from torch import nn
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.traj_dit import TrajDiT, TrajParams
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from models.modules.sampling import prepare_ids, get_schedule

class TrainTransformersDiT(nn.Module):
    def __init__(
        self,
        args,
        local_rank=-1,
        load_path=None,
        condition_frames=3,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size ** 2
        self.image_size = self.args.image_size
        self.traj_len = self.args.traj_len
        self.h, self.w = (self.image_size[0]//(self.args.downsample_size*self.args.patch_size),  self.image_size[1]//(self.args.downsample_size*self.args.patch_size))
        self.pkeep = args.pkeep

        self.img_token_size = self.h * self.w
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_x_bound = self.args.pose_x_bound
        self.pose_y_bound = self.args.pose_y_bound
        self.yaw_bound = self.args.yaw_bound

        self.pose_x_range = getattr(args, 'pose_x_range', 8.0)
        self.pose_y_range = getattr(args, 'pose_y_range', 1.0)
        self.pose_yaw_range = getattr(args, 'pose_yaw_range', 16.0)

        self.pose_token_size = 2 * self.args.block_size
        self.yaw_token_size = 1 * self.args.block_size
        self.traj_token_size = self.pose_token_size + self.yaw_token_size
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size
        self.token_size_dict = {
            'img_tokens_size': self.img_token_size,
            'pose_tokens_size': self.pose_token_size,
            'yaw_token_size': self.yaw_token_size,
            'total_tokens_size': self.total_token_size
        }

        self.model = SpatialTemporalTransformer(
            block_size=condition_frames*(self.total_token_size),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w),

            local_rank=local_rank,
            condition_frames=self.condition_frames,
            token_size_dict=self.token_size_dict,
            vae_emb_dim = self.vae_emb_dim,
            temporal_block=self.args.block_size
        )
        self.model.cuda()

        self.dit = FluxDiT(FluxParams(
            in_channels=self.vae_emb_dim,
            out_channels=self.vae_emb_dim,
            vec_in_dim=args.n_embd*(self.total_token_size-self.img_token_size),
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit,
            mlp_ratio=4.0,
            num_heads=args.n_head_dit,
            depth=args.n_layer[1],
            depth_single_blocks=args.n_layer[2],
            axes_dim=args.axes_dim_dit,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ))
        self.dit.cuda()

        self.traj_dit = TrajDiT(TrajParams(
            in_channels=self.traj_token_size,
            out_channels=self.traj_token_size,
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit_traj,
            mlp_ratio=4.0,
            num_heads=args.n_head_dit_traj,
            depth=args.n_layer_traj[0],
            depth_single_blocks=args.n_layer_traj[1],
            axes_dim=args.axes_dim_dit_traj,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ))
        self.traj_dit.cuda()

        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, self.traj_ids = prepare_ids(bs, self.h, self.w, self.total_token_size, self.traj_len)
        self.lambda_yaw_pose = self.args.lambda_yaw_pose
        if load_path is not None:

            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]
            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.model.'+k]
            self.model.load_state_dict(model_state_dict)
            traj_dit_state_dict = self.traj_dit.state_dict()
            if any(k.startswith('module.traj_dit.') for k in state_dict.keys()):
                for k in traj_dit_state_dict.keys():
                    traj_dit_state_dict[k] = state_dict['module.traj_dit.'+k]
                self.traj_dit.load_state_dict(traj_dit_state_dict)
            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                dit_state_dict[k] = state_dict['module.dit.'+k]
            self.dit.load_state_dict(dit_state_dict)
            print(f"Successfully load model from {load_path}")
    def denormalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = (traj_targets[..., 0:1] + 1) * self.pose_x_bound / 2
        traj_targets[..., 1:2] *= self.pose_y_bound
        traj_targets[..., 2:3] *= self.yaw_bound
        return traj_targets
    @torch.no_grad()
    def step_eval(self, latents, rel_pose, rel_yaw, **kwargs):
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size, x_range=self.pose_x_range, y_range=self.pose_y_range)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size, yaw_range=self.pose_yaw_range)
        features, _ = self.model.evaluate(latents, pose_total, yaw_total, sample_last=True)
        _, cond_ids, traj_ids = prepare_ids(features.shape[0], self.h, self.w, self.total_token_size, self.traj_len)
        timesteps = get_schedule(int(self.args.num_sampling_steps_traj), self.traj_len)
        noise = torch.randn(features.shape[0], self.traj_len, self.traj_token_size).to(features)
        trajectory = self.traj_dit.sample(noise, traj_ids, features, cond_ids, timesteps, deterministic=True, num_samples=1)
        limit = float(self.args.trajectory_yaw_clip)
        trajectory[..., 2:3] = trajectory[..., 2:3].clamp(-limit, limit)
        return self.denormalize_traj(trajectory), None
