from __future__ import annotations

import logging
from typing import Optional, Tuple
import torch
import numpy as np
from PIL import Image

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from utils.config_utils import Config
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.preprocess import get_rel_pose

logger = logging.getLogger(__name__)

def _build_se2_matrix(x: float, y: float, yaw: float) -> np.ndarray:

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array([[cos_yaw, -sin_yaw,0.0, x],
                     [sin_yaw, cos_yaw,0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

def _normalize_images(images:torch.Tensor) -> torch.Tensor:

    images = images/255.0
    return (images - 0.5) / 0.5

def _resample_trajectory(poses:np.ndarray,target_num_poses:int) -> np.ndarray:
    if poses.shape[0] == target_num_poses:
        return poses
    if poses.shape[0] < 2:
        return np.repeat(poses[:1], target_num_poses, axis=0)

    src = np.linspace(0, 1, poses.shape[0])
    dst = np.linspace(0, 1, target_num_poses)

    x_interp=np.interp(dst, src, poses[:, 0])
    y_interp=np.interp(dst, src, poses[:, 1])
    yaw_unwrapped=np.unwrap(poses[:, 2])
    yaw_interp=np.interp(dst, src, yaw_unwrapped)
    yaw_interp=(yaw_interp+np.pi)%(2*np.pi)-np.pi

    return np.stack([x_interp, y_interp, yaw_interp], axis=1)

class EponaAgent(AbstractAgent):

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        vae_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        use_amp: bool = True,
        output_time_horizon:float=4.0,
        output_interval:float=0.5,
        condition_frames:Optional[int]=None,
        traj_len_override:Optional[int]=None,
        image_size:Optional[Tuple[int, int]]=None,
    ):

        super().__init__(requires_scene=False)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.vae_checkpoint_path = vae_checkpoint_path
        self.device = device
        self.use_amp = use_amp
        self.output_time_horizon = output_time_horizon
        self.output_interval = output_interval
        self.condition_frames_override = condition_frames
        self.traj_len_override = traj_len_override
        self.image_size_override = image_size
        self._initialized=False
        self._model:Optional[TrainTransformersDiT]=None
        self._tokenizer:Optional[VAETokenizer]=None
        self._condition_frames:Optional[int]=None
        self._traj_len:Optional[int]=None
        self._image_size:Optional[Tuple[int, int]]=None
        self._device:Optional[torch.device]=None

    def name(self) -> str:
        return "epona_agent"

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def initialize(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for EponaAgent.")
        if self.device != "cuda":
            raise ValueError("EponaAgent only supports CUDA because the model uses .cuda() internally.")

        cfg = Config.fromfile(self.config_path)
        if not hasattr(cfg, "batch_size"):
            cfg.batch_size = 1

        if self.vae_checkpoint_path:
            cfg.vae_ckpt = self.vae_checkpoint_path

        if self.condition_frames_override is not None:
            cfg.condition_frames = self.condition_frames_override

        if self.traj_len_override is not None:
            cfg.traj_len = self.traj_len_override

        if self.image_size_override is not None:
            cfg.image_size = tuple(self.image_size_override)

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for EponaAgent inference.")

        self._device = torch.device(self.device)
        self._condition_frames = int(cfg.condition_frames)
        self._traj_len = int(cfg.traj_len)
        self._image_size = tuple(cfg.image_size)

        self._model = TrainTransformersDiT(
            cfg,
            load_path=self.checkpoint_path,
            local_rank=0,
            condition_frames=self._condition_frames,
        )
        self._model.eval()
        self._tokenizer = VAETokenizer(cfg, local_rank=0)
        self._initialized = True

    def compute_trajectory(
        self, agent_input: AgentInput
    ) -> Trajectory:
        if not self._initialized:
            self.initialize()

        assert self._model is not None
        assert self._tokenizer is not None
        assert self._condition_frames is not None
        assert self._traj_len is not None
        assert self._image_size is not None
        assert self._device is not None

        history_len = len(agent_input.ego_statuses)
        if history_len == 0:
            logger.warning("Empty agent input. Returning zero trajectory.")
            return self._build_fallback_trajectory()

        start_idx = max(0, history_len - self._condition_frames)
        cameras = agent_input.cameras[start_idx:]
        ego_statuses = agent_input.ego_statuses[start_idx:]

        if len(cameras) < self._condition_frames:
            pad_count = self._condition_frames - len(cameras)
            cameras = [cameras[0]] * pad_count + cameras
            ego_statuses = [ego_statuses[0]] * pad_count + ego_statuses

        images = []
        for camera in cameras:
            image = camera.cam_f0.image
            if image is None:

                image = np.zeros((self._image_size[0], self._image_size[1], 3), dtype=np.uint8)
            pil = Image.fromarray(image)
            pil = pil.resize((self._image_size[1], self._image_size[0]), Image.BICUBIC)
            images.append(np.array(pil, dtype=np.uint8))

        images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float()
        images = _normalize_images(images).unsqueeze(0).to(self._device)

        pose_mats = []
        for status in ego_statuses:
            x, y, yaw = status.ego_pose
            pose_mats.append(_build_se2_matrix(x, y, yaw))

        pose_mats.append(pose_mats[-1].copy())
        pose_tensor = torch.from_numpy(np.stack(pose_mats, axis=0)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            rel_pose, rel_yaw = get_rel_pose(pose_tensor)
            latents = self._tokenizer.encode_to_z(images)
            use_amp = self.use_amp and self._device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predict_traj, _ = self._model.step_eval(
                    latents,
                    rel_pose[:, : self._condition_frames + 1],
                    rel_yaw[:, : self._condition_frames + 1],
                )

        traj = predict_traj[0].detach().cpu().numpy()
        if traj.shape[-1] > 3:

            traj = traj[..., :3]

        traj[:, 2] = np.deg2rad(traj[:, 2])
        output_num_poses = int(round(self.output_time_horizon / self.output_interval))
        traj = _resample_trajectory(traj, output_num_poses)
        trajectory_sampling = TrajectorySampling(
            time_horizon=self.output_time_horizon,
            interval_length=self.output_interval,
        )
        return Trajectory(traj.astype(np.float32), trajectory_sampling)

    def _build_fallback_trajectory(self) -> Trajectory:
        output_num_poses = int(round(self.output_time_horizon / self.output_interval))
        traj = np.zeros((output_num_poses, 3), dtype=np.float32)
        trajectory_sampling = TrajectorySampling(
            time_horizon=self.output_time_horizon,
            interval_length=self.output_interval,
        )
        return Trajectory(traj, trajectory_sampling)
