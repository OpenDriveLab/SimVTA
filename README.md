# SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

[![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white&labelColor=555)](https://arxiv.org/abs/2610.13781) [![Project Page](https://img.shields.io/badge/Project_Page-8A2BE2?logo=googlechrome&logoColor=white&labelColor=555)](https://opendrivelab.com/SimVTA/) [![Checkpoints](https://img.shields.io/badge/-Checkpoints-FFD21E?logo=huggingface&logoColor=white&labelColor=555)](https://huggingface.co/OpenDriveLab-org/SimVTA)

> [Changhui Jing](https://scholar.google.com/citations?user=B4Nu6pAAAAAJ), [Haochen Tian](https://scholar.google.com/citations?user=DoyuNwEAAAAJ), [Hongchen Li](https://scholar.google.com/citations?user=QO6p6OgAAAAJ), [Tianyu Li](https://sephyli.github.io/), [Caojun Wang](https://scholar.google.com/citations?user=35xHlDUAAAAJ),
> [Zihao Zhang](https://scholar.google.com/citations?user=R2Brvm4AAAAJ), [Yanhao Liu](https://scholar.google.com/citations?user=dYicC-4AAAAJ), [Chengen Xie](https://scholar.google.com/citations?user=q0W3n-cAAAAJ), [Bailin Li](https://www.linkedin.com/in/libailin/), [Hao Zhao](https://sites.google.com/view/fromandto), [Zhenjie Yang](https://scholar.google.com/citations?user=jVlRiUEAAAAJ)

> - 📧 Primary Contact: Changhui Jing ([changhui.jing@opendrivelab.com](mailto:changhui.jing@opendrivelab.com))
> - 🖊️ Joint effort by Shanghai Innovation Institute, OpenDriveLab at HKU, Li Auto Inc., Tongji University, and Tsinghua University.

---

![From logged futures to candidate-matched supervision](https://ik.imagekit.io/szpk/SimVTA/overview.png)

## Highlights

- 🚗 **Simulation-grounded alignment:** Render matched future videos for alternative candidate trajectories in reconstructed real-world scenes.
- 🔍 **Selective video supervision:** Prioritize higher-error futures among high-reward candidates and jointly train trajectory planning and video prediction.
- 📈 **Improved planning:** Achieve 91.3 PDMS on NAVSIM v1, 88.2 EPDMS on NAVSIM v2, and 38.8 average HD-Score on HUGSIM, with direct trajectory prediction at inference.

## News

- `[2026/10/23]` We released the [checkpoints](https://huggingface.co/OpenDriveLab-org/SimVTA) of SimVTA.
- `[2026/10/12]` We released our [paper](https://arxiv.org/abs/2610.13781) on arXiv.

## 📋 TODO List

- [x] Results release.
- [x] Checkpoint release.
- [x] Inference and evaluation code.
- [ ] Training code.


## Getting Started

**1. Clone SimVTA**

```bash
git clone https://github.com/OpenDriveLab/SimVTA.git
cd SimVTA
```

**2. Create Environment**

```bash
conda create -n simvta python=3.10 -y
conda activate simvta
pip install -r requirements.txt
pip install -e .
```

Use a CUDA-compatible PyTorch installation and a GPU with bfloat16 support. NAVSIM v1 is included in `navsim/`.

**3. Data Preparation**

Prepare the NAVSIM v1 OpenScene navtest logs, sensor blobs, and nuPlan maps following the [NAVSIM data instructions](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md). Organize the files as follows:

```text
SimVTA/
├── checkpoints/
│   ├── simvta.pkl
│   └── dcae_td_20000.pkl
├── datasets/
│   ├── maps/
│   │   └── nuplan-maps-v1.0/
│   └── openscene/
│       ├── navsim_logs/
│       │   └── test/
│       └── sensor_blobs/
└── exp/
    └── metric_cache/
```

**4. Download Checkpoints**

Download `simvta.pkl` from [OpenDriveLab-org/SimVTA](https://huggingface.co/OpenDriveLab-org/SimVTA) and the temporal-aware DCAE checkpoint `dcae_td_20000.pkl` from [Kevin-thu/Epona](https://huggingface.co/Kevin-thu/Epona). Place both files in `checkpoints/`:

```bash
mkdir -p checkpoints
wget -c https://huggingface.co/OpenDriveLab-org/SimVTA/resolve/main/simvta.pkl -O checkpoints/simvta.pkl
wget -c https://huggingface.co/Kevin-thu/Epona/resolve/main/dcae_td_20000.pkl -O checkpoints/dcae_td_20000.pkl
```

The DCAE checkpoint encodes the input camera images into latents. Use Epona's temporal-aware checkpoint for this configuration.

**5. NAVSIM v1 Evaluation**

```bash
bash run/test_navsim.sh
```

## Overview

**SimVTA** is a simulation-grounded post-training framework for driving world action models. It extends supervision beyond logged trajectories using a 3D Gaussian Splatting simulator, combining trajectory GRPO with a flow-matching video loss that updates shared representations.

Reported results in the paper:

| Benchmark | Metric | SimVTA |
| --- | --- | --- |
| NAVSIM v1 | PDMS | **91.3** (+5.1 over the 86.2 initialization) |
| NAVSIM v2 | EPDMS | **88.2** |
| HUGSIM | Average HD-Score | **38.8** |

## Matched Futures for Alternative Actions

![Simulator-rendered futures for straight driving and lane changes](https://ik.imagekit.io/szpk/SimVTA/matched-futures.png)

**Simulator-rendered futures for going straight (A), changing lanes left (B), and changing lanes right (C).**

## Video-Trajectory Alignment

![Video-trajectory alignment before and after post-training](https://ik.imagekit.io/szpk/SimVTA/video-trajectory-alignment.png?updatedAt=1792686923154)

**Video-trajectory alignment: World Engine renderings compared with SimVTA and Epona predictions.**

## Acknowledgements

We thank the authors and open-source contributors of the following projects for providing the models, simulation tools, and evaluation benchmarks that support this work:

- [Epona](https://github.com/Kevin-thu/Epona) | [World Engine](https://github.com/OpenDriveLab/WorldEngine) | [NAVSIM](https://github.com/autonomousvision/navsim) | [HUGSIM](https://github.com/hyzhou404/HUGSIM)

We also thank the broader open-source community for its contributions to driving world models, autonomous driving, and 3D Gaussian Splatting.

## License and Citation

All content in this repository is licensed under the
[Apache-2.0 license](LICENSE).

If our paper or code helps your research, please consider citing
our work and giving this repository a star.