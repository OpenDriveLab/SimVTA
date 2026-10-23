from pathlib import Path
from typing import List, Tuple
import yaml
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

ROOT = Path(__file__).resolve().parents[1]

def _build_scene_filter(cfg) -> SceneFilter:
    num_history_frames = cfg.navsim_history_frames or cfg.condition_frames
    required_future = int(round(cfg.output_time_horizon / cfg.output_interval))
    num_future_frames = max(cfg.navsim_future_frames or cfg.traj_len, required_future)

    navtest_config_path = ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    log_names, tokens = None, None
    if navtest_config_path.exists():
        with open(navtest_config_path, 'r') as f:
            navtest_cfg = yaml.safe_load(f)
        log_names = navtest_cfg.get('log_names')
        tokens = navtest_cfg.get('tokens')

    return SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=cfg.frame_interval,
        max_scenes=cfg.max_scenes,
        log_names=log_names,
        tokens=tokens,
    )

def _collect_tokens(cfg, scene_filter: SceneFilter) -> Tuple[List[str], int]:
    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = scene_loader.tokens
    start_id = cfg.start_id
    end_id = min(cfg.end_id, len(tokens))
    return tokens[start_id:end_id], len(tokens)
