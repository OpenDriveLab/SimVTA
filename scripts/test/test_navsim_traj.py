import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from utils.config_utils import Config
from utils.navsim_utils import _build_scene_filter

from navsim.agents.epona.epona_agent import EponaAgent
from navsim.common.dataloader import SceneLoader

def add_arguments():
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    default_navsim_log_path = (
        os.path.join(openscene_root, "navsim_logs/test") if openscene_root else ""
    )
    default_sensor_blobs_path = (
        os.path.join(openscene_root, "sensor_blobs") if openscene_root else ""
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video_path", type=str, default="test_videos")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--config", default="configs/navsim_v1.py", type=str)
    parser.add_argument("--resume_path", default=None, type=str, help="pretrained path")
    parser.add_argument("--vae_ckpt", default=None, type=str, help="optional VAE checkpoint override")
    parser.add_argument("--condition_frames", type=int, default=4)
    parser.add_argument("--traj_len", type=int, default=8)

    parser.add_argument("--navsim_log_path", type=str, default=default_navsim_log_path)
    parser.add_argument("--sensor_blobs_path", type=str, default=default_sensor_blobs_path)
    parser.add_argument("--navsim_history_frames", type=int, default=None)
    parser.add_argument("--navsim_future_frames", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=99999)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_time_horizon", type=float, default=4.0)
    parser.add_argument("--output_interval", type=float, default=0.5)

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    overrides = {}
    for key, value in vars(args).items():
        if value is None and hasattr(cfg, key):
            continue
        overrides[key] = value
    cfg.merge_from_dict(overrides)
    return cfg

args = add_arguments()

seed = os.environ.get("EPONA_EVAL_SEED", "")
if seed:
    seed = int(seed)
    print(f"Global evaluation seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

def main(cfg):
    if not cfg.navsim_log_path or not cfg.sensor_blobs_path:
        raise ValueError("navsim_log_path and sensor_blobs_path are required.")
    if not cfg.resume_path:
        raise ValueError("resume_path is required for inference.")
    save_path = os.path.join(cfg.save_video_path, cfg.exp_name)
    os.makedirs(save_path, exist_ok=True)

    agent = EponaAgent(
        config_path=cfg.config,
        checkpoint_path=cfg.resume_path,
        vae_checkpoint_path=getattr(cfg, "vae_ckpt", None),
        device=cfg.device,
        use_amp=not cfg.no_amp,
        output_time_horizon=cfg.output_time_horizon,
        output_interval=cfg.output_interval,
        condition_frames=cfg.condition_frames,
        traj_len_override=cfg.traj_len,
        image_size=cfg.image_size,
    )
    agent.initialize()
    scene_filter = _build_scene_filter(cfg)
    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    all_tokens = scene_loader.tokens
    tokens = all_tokens[cfg.start_id:min(cfg.end_id, len(all_tokens))]
    if not tokens:
        raise RuntimeError("No scenes found for the requested range")

    for token in tqdm(tokens, desc="NAVSIM inference"):
        agent_input = scene_loader.get_agent_input_from_token(token)
        prediction = agent.compute_trajectory(agent_input)
        np.save(os.path.join(save_path, f"{token}.npy"), prediction.poses)

if __name__ == "__main__":
    main(args)
