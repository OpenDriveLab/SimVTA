<div align="center">

# SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

Changhui Jing<sup>1,2,4</sup>, Haochen Tian<sup>2</sup>, Hongchen Li<sup>1,2,4</sup>, Tianyu Li<sup>1</sup>, Caojun Wang<sup>1</sup>,<br>
Zihao Zhang<sup>2</sup>, Yanhao Liu<sup>1</sup>, Chengen Xie<sup>1</sup>, Bailin Li<sup>3</sup>, Hao Zhao<sup>5</sup>, Zhenjie Yang<sup>2,‡</sup>

<sup>1</sup> Shanghai Innovation Institute · <sup>2</sup> OpenDriveLab at The University of Hong Kong<br>
<sup>3</sup> Li Auto Inc. · <sup>4</sup> Tongji University · <sup>5</sup> Tsinghua University

</div>

> 📧 Primary Contact: Changhui Jing ([changhui.jing@opendrivelab.com](mailto:changhui.jing@opendrivelab.com))  
> ‡ Corresponding Author: Zhenjie Yang

---

<p align="center">
  <img src="assets/overview.png" width="100%" alt="From logged futures to candidate-matched supervision">
</p>

**From logged futures to candidate-matched supervision.** WAM pretraining uses the ground-truth trajectory and its logged future video. Trajectory GRPO explores candidate trajectories using driving rewards, without matched video supervision. World Engine renders reconstructed scenes along selected candidate paths, providing matched supervision independent of the model's predictions.

## Overview

**SimVTA** is a simulation-grounded post-training framework that combines trajectory optimization with video supervision for driving world action models. Driving logs provide future videos only for recorded trajectories; SimVTA supplies matched visual supervision for alternative actions by rendering reconstructed real-world scenes along high-reward candidate paths with a 3D Gaussian Splatting simulator.

Under a limited rendering budget, SimVTA prioritizes higher-error futures among the rendered candidates. It jointly optimizes trajectory planning with Group Relative Policy Optimization (GRPO) and video prediction with a flow-matching loss that updates shared representations. At inference time, the model predicts trajectories directly, without simulator calls.

Reported results in the paper:

| Benchmark | Metric | SimVTA |
| --- | --- | --- |
| NAVSIM v1 | PDMS | **91.3** (+5.1 over the 86.2 initialization) |
| NAVSIM v2 | EPDMS | **88.2** |
| HUGSIM | Average HD-Score | **37.3** |

## Matched Futures for Alternative Actions

<p align="center">
  <img src="assets/matched-futures.png" width="100%" alt="Simulator-rendered futures for straight driving and lane changes">
</p>

**Matched futures for alternative actions.** Left: shared observation history and executed paths for going straight (A), changing lanes to the left (B), and changing lanes to the right (C). Right: corresponding future frames rendered by the simulator. Colors match the paths and image rows.

## Video-Trajectory Alignment

<p align="center">
  <img src="assets/video-trajectory-alignment.png" width="100%" alt="Video-trajectory alignment before and after post-training">
</p>

**Video-trajectory alignment before and after post-training.** Left: observation history and trajectory in bird's-eye view. Right: World Engine renderings and predictions from SimVTA and Epona.
