import argparse
import hashlib
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from utils.config_utils import Config
from utils.navsim_utils import _build_scene_filter, _collect_tokens

from navsim.common.dataclasses import Trajectory

def _run_command(cmd: List[str], env=None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)

def _format_hydra_list(items: List[str]) -> str:
    return "[" + ",".join(f"'{item}'" for item in items) + "]"

def _safe_filename_component(value: str) -> str:

    return "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in str(value)
    ).strip("-") or "unnamed"

def _build_submission(cfg, tokens: List[str], submission_path: Path) -> None:
    preds_dir = Path(cfg.save_video_path) / cfg.exp_name
    sampling = TrajectorySampling(
        time_horizon=cfg.output_time_horizon,
        interval_length=cfg.output_interval,
    )
    predictions = {}
    for token in tokens:
        pred_file = preds_dir / f"{token}.npy"
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        poses = np.load(pred_file)
        if poses.shape[0] != sampling.num_poses:
            raise ValueError(f"Unexpected trajectory shape for {token}: {poses.shape}")
        predictions[token] = Trajectory(poses.astype(np.float32), sampling)
    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "predictions": [predictions],
    }
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with open(submission_path, "wb") as f:
        pickle.dump(submission, f)
    print(f"Saved submission to {submission_path}")

def add_arguments():
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    default_navsim_log_path = (
        os.path.join(openscene_root, "navsim_logs/test") if openscene_root else ""
    )
    default_sensor_blobs_path = (
        os.path.join(openscene_root, "sensor_blobs") if openscene_root else ""
    )
    default_exp_root = os.environ.get("NAVSIM_EXP_ROOT", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/navsim_v1.py")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--condition_frames", type=int, default=4)
    parser.add_argument("--traj_len", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_time_horizon", type=float, default=4.0)
    parser.add_argument("--output_interval", type=float, default=0.5)
    parser.add_argument("--save_video_path", type=str, default="test_videos")
    parser.add_argument("--navsim_log_path", type=str, default=default_navsim_log_path)
    parser.add_argument("--sensor_blobs_path", type=str, default=default_sensor_blobs_path)
    parser.add_argument("--navsim_history_frames", type=int, default=None)
    parser.add_argument("--navsim_future_frames", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=99999)
    parser.add_argument("--navsim_exp_root", type=str, default=default_exp_root)
    parser.add_argument("--metric_cache_path", type=str, default=None)
    parser.add_argument("--submission_path", type=str, default=None)
    parser.add_argument("--score_output_dir", type=str, default=None)
    parser.add_argument("--team_name", type=str, default="epona")
    parser.add_argument("--authors", type=str, default="epona")
    parser.add_argument("--email", type=str, default="epona@example.com")
    parser.add_argument("--institution", type=str, default="epona")
    parser.add_argument("--country", type=str, default="china")

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    overrides = {}
    for key, value in vars(args).items():
        if value is None and hasattr(cfg, key):
            continue
        overrides[key] = value
    cfg.merge_from_dict(overrides)
    return cfg

def _build_inference_cmd(cfg, gpu_start_id: int, gpu_end_id: int, device: str) -> List[str]:

    cmd = [
        sys.executable,
        "scripts/test/test_navsim_traj.py",
        "--exp_name", cfg.exp_name,
        "--config", cfg.config,
        "--resume_path", cfg.resume_path,
        "--navsim_log_path", cfg.navsim_log_path,
        "--sensor_blobs_path", cfg.sensor_blobs_path,
        "--frame_interval", str(cfg.frame_interval),
        "--start_id", str(gpu_start_id),
        "--end_id", str(gpu_end_id),
        "--device", device,
        "--save_video_path", cfg.save_video_path,
        "--output_time_horizon", str(cfg.output_time_horizon),
        "--output_interval", str(cfg.output_interval),
        "--condition_frames", str(cfg.condition_frames),
        "--traj_len", str(cfg.traj_len),
    ]
    if cfg.no_amp:
        cmd.append("--no_amp")
    if cfg.vae_ckpt:
        cmd.extend(["--vae_ckpt", cfg.vae_ckpt])
    if cfg.navsim_history_frames is not None:
        cmd.extend(["--navsim_history_frames", str(cfg.navsim_history_frames)])
    if cfg.navsim_future_frames is not None:
        cmd.extend(["--navsim_future_frames", str(cfg.navsim_future_frames)])
    if cfg.max_scenes is not None:
        cmd.extend(["--max_scenes", str(cfg.max_scenes)])
    return cmd

def main(cfg):
    if not cfg.navsim_log_path or not cfg.sensor_blobs_path:
        raise ValueError("navsim_log_path and sensor_blobs_path are required.")
    if not cfg.resume_path:
        raise ValueError("resume path is required.")
    if not Path(cfg.navsim_log_path).exists():
        raise FileNotFoundError(f"navsim_log_path does not exist: {cfg.navsim_log_path}")
    if not Path(cfg.sensor_blobs_path).exists():
        raise FileNotFoundError(f"sensor_blobs_path does not exist: {cfg.sensor_blobs_path}")
    if not os.environ.get("NUPLAN_MAPS_ROOT"):
        raise EnvironmentError("NUPLAN_MAPS_ROOT is required for metric caching.")

    exp_root = cfg.navsim_exp_root or "navsim_exp"
    metric_cache_path = cfg.metric_cache_path or os.path.join(exp_root, "metric_cache")
    submission_path = cfg.submission_path or os.path.join(
        exp_root, "submissions", cfg.exp_name, "submission.pkl"
    )
    score_output_dir = cfg.score_output_dir or os.path.join(exp_root, "pdm_scores")
    scene_filter = _build_scene_filter(cfg)
    tokens, total_tokens = _collect_tokens(cfg, scene_filter)
    if not tokens:
        raise RuntimeError("No tokens found for the given scene filter.")

    cmd = _build_inference_cmd(cfg, cfg.start_id, cfg.end_id, cfg.device)
    _run_command(cmd)

    _build_submission(cfg, tokens, Path(submission_path))

    node_id = int(os.environ.get("NODE_RANK", 0))
    metric_cache_path_obj = Path(metric_cache_path)
    metadata_csv_path = metric_cache_path_obj / "metadata" / f"metric_cache_metadata_node_{node_id}.csv"

    if not metadata_csv_path.exists():
        metric_cmd = [
            sys.executable,
            "navsim/planning/script/run_metric_caching.py",
            "train_test_split=navtest",
            "worker=sequential",
            f"navsim_log_path={cfg.navsim_log_path}",
            f"cache.cache_path={metric_cache_path}",
            f"train_test_split.scene_filter.num_history_frames={scene_filter.num_history_frames}",
            f"train_test_split.scene_filter.num_future_frames={scene_filter.num_future_frames}",
            f"train_test_split.scene_filter.frame_interval={scene_filter.frame_interval}",
            "train_test_split.scene_filter.log_names=null",
        ]
        if cfg.max_scenes is not None:
            metric_cmd.append(f"train_test_split.scene_filter.max_scenes={scene_filter.max_scenes}")
        if cfg.start_id != 0 or cfg.end_id < total_tokens:
            metric_cmd.append(
                f"train_test_split.scene_filter.tokens={_format_hydra_list(tokens)}"
            )
        _run_command(metric_cmd)

    resume_path = Path(cfg.resume_path)
    ckpt_tag = _safe_filename_component(resume_path.stem)
    train_name = _safe_filename_component(resume_path.parent.name)
    exp_name = _safe_filename_component(cfg.exp_name)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")
    score_dir = Path(score_output_dir)
    score_dir.mkdir(parents=True, exist_ok=True)

    run_digest = hashlib.sha256(
        f"{exp_name}/{train_name}/{ckpt_tag}".encode("utf-8")
    ).hexdigest()[:16]
    score_csv_path = score_dir / (
        f"{timestamp}.{ckpt_tag[:48]}.{run_digest}.pid{os.getpid()}.csv"
    )

    epona_script = ROOT / "navsim/planning/script/run_pdm_score_from_submission.py"
    score_cmd = [
        sys.executable,
        str(epona_script),
        "train_test_split=navtest",
        f"submission_file_path={submission_path}",
        f"metric_cache_path={metric_cache_path}",
        f"output_dir={score_output_dir}",
        f"+output_file_path={score_csv_path}",
    ]

    subprocess.run(score_cmd, cwd=str(ROOT), check=True)
    if not score_csv_path.is_file():
        raise RuntimeError(f"PDM scorer did not create expected CSV: {score_csv_path}")
    print(f"PDM scores saved to {score_csv_path}")

if __name__ == "__main__":
    main(add_arguments())
