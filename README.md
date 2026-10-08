# SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

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

## 📋 TODO List

- [x] Results release.
- [ ] More visualization results.
- [ ] Code release.
- [ ] Checkpoints release.

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

![Video-trajectory alignment before and after post-training](https://ik.imagekit.io/szpk/SimVTA/video-trajectory-alignment.png)

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
