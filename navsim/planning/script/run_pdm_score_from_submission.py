from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path
from dataclasses import asdict
import logging
import pickle
import os
import multiprocessing as mp

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"


def run_pdm_score(
    submission_file_path: Path,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
    num_workers: int = 1,
) -> pd.DataFrame:
    """
    Evaluate submission file with PDM score.
    :param submission_file_path: path to submission pickle file
    :param simulator: internal simulator object of PDM
    :param scorer: internal scoring objected in PDM
    :param metric_cache_path: path to metric cache
    :param num_workers: CPU processes; 1 retains serial execution.
    :return: pandas data frame with PDMS results.
    """
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    with open(submission_file_path, "rb") as f:
        agent_output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]

    assert len(agent_output_list) == 1, "Multi-seed evaluation currently not supported in run_pdm_score!"
    agent_output = agent_output_list[0]


    submission_tokens = list(agent_output.keys())
    logger.info(f"Processing {len(submission_tokens)} tokens from submission file (out of {len(metric_cache_loader.tokens)} total tokens in metric cache)")

    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    tasks = ((token, agent_output[token]) for token in submission_tokens)
    if num_workers == 1:
        _init_score_worker(metric_cache_loader, simulator, scorer)
        score_rows = list(tqdm(map(_score_token, tasks), total=len(submission_tokens), desc="Compute PDM-Score"))
    else:
        # Spawn independent simulator/scorer state. Ordered imap preserves the
        # serial row order and therefore the floating-point reduction order.
        with mp.get_context("spawn").Pool(
            min(num_workers, max(1, len(submission_tokens))),
            initializer=_init_score_worker,
            initargs=(metric_cache_loader, simulator, scorer),
        ) as pool:
            score_rows = list(tqdm(pool.imap(_score_token, tasks, chunksize=1),
                                   total=len(submission_tokens), desc="Compute PDM-Score"))
    pdm_score_df = pd.DataFrame(score_rows)
    if not pdm_score_df["valid"].all():
        raise RuntimeError("Evaluation failed for one or more tokens; refusing to report a partial average")
    return pdm_score_df


def _init_score_worker(loader, simulator, scorer):
    global _worker_loader, _worker_simulator, _worker_scorer
    _worker_loader, _worker_simulator, _worker_scorer = loader, simulator, scorer


def _score_token(task):
    token, trajectory = task
    score_row = {"token": token, "valid": True}
    try:
        result = pdm_score(
            metric_cache=_worker_loader.get_from_token(token),
            model_trajectory=trajectory,
            future_sampling=_worker_simulator.proposal_sampling,
            simulator=_worker_simulator,
            scorer=_worker_scorer,
        )
        score_row.update(asdict(result))
    except Exception:
        logger.exception("Agent failed for token %s", token)
        score_row["valid"] = False
    return score_row


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS from submission pickle.
    :param cfg: omegaconf dictionary
    """
    submission_file_path = Path(cfg.submission_file_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    build_logger(cfg)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    pdm_score_df = run_pdm_score(
        submission_file_path=submission_file_path,
        simulator=simulator,
        scorer=scorer,
        metric_cache_path=metric_cache_path,
        num_workers=int(os.environ.get("PDMS_SCORE_WORKERS", "1")),
    )

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    output_file_path = getattr(cfg, "output_file_path", None)
    if output_file_path:
        csv_path = Path(output_file_path)
    else:
        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        csv_path = save_path / f"{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pdm_score_df.to_csv(csv_path)
    logger.info("PDM scores saved to: %s", csv_path)


if __name__ == "__main__":
    main()
