# SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

> [Changhui Jing](https://scholar.google.com/citations?hl=en&user=B4Nu6pAAAAAJ), [Haochen Tian](https://github.com/hctian713), Hongchen Li, [Tianyu Li](https://sephyli.github.io/), Caojun Wang,
> [Zihao Zhang](https://github.com/Zizizi-hao), Yanhao Liu, Chengen Xie, Bailin Li, Hao Zhao, [Zhenjie Yang](https://scholar.google.com/citations?hl=en&user=jVlRiUEAAAAJ)

> - 📧 Primary Contact: Changhui Jing ([changhui.jing@opendrivelab.com](mailto:changhui.jing@opendrivelab.com))
> - 🖊️ Joint effort by Shanghai Innovation Institute, OpenDriveLab at HKU, Li Auto Inc., Tongji University, and Tsinghua University.
> - Corresponding Author: Zhenjie Yang

---

![From logged futures to candidate-matched supervision](assets/overview.png)

**From logged futures to candidate-matched supervision.** WAM pretraining uses the ground-truth trajectory and its logged future video. Trajectory GRPO explores candidate trajectories using driving rewards, without matched video supervision. World Engine renders reconstructed scenes along selected candidate paths, providing matched supervision independent of the model's predictions.

## Highlights

- 🚗 **Simulation-grounded alignment:** Render matched future videos for alternative candidate trajectories in reconstructed real-world scenes.
- 🔍 **Selective video supervision:** Prioritize higher-error futures among high-reward candidates and jointly train trajectory planning and video prediction.
- 📈 **Improved planning:** Achieve 91.3 PDMS on NAVSIM v1, 88.2 EPDMS on NAVSIM v2, and 37.3 average HD-Score on HUGSIM, with direct trajectory prediction at inference.

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

![Simulator-rendered futures for straight driving and lane changes](assets/matched-futures.png)

**Matched futures for alternative actions.** Left: shared observation history and executed paths for going straight (A), changing lanes to the left (B), and changing lanes to the right (C). Right: corresponding future frames rendered by the simulator. Colors match the paths and image rows.

## Video-Trajectory Alignment

![Video-trajectory alignment before and after post-training](assets/video-trajectory-alignment.png)

**Video-trajectory alignment before and after post-training.** Left: observation history and trajectory in bird's-eye view. Right: World Engine renderings and predictions from SimVTA and Epona.

## Acknowledgements

We thank the authors and open-source contributors of the following projects for providing the models, simulation tools, and evaluation benchmarks that support this work:

- Epona | World Engine | [NAVSIM](https://github.com/autonomousvision/navsim) | [HUGSIM](https://github.com/hyzhou404/HUGSIM)

We also thank the broader open-source community for its contributions to driving world models, autonomous driving, and 3D Gaussian Splatting.
