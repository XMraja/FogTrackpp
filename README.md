# FogTrack++: Adaptive Vision-Language Inheritance Learning for Multi-Vessel Tracking Onboard EO Gimbal System

**Zijie Zhang, Changhong Fu\*, Mengyuan Li, Liangliang Yao, Haobo Zuo, Guangze Zheng, Shan An**

\* Corresponding author.

## Abstract

Unmanned aerial vehicle (UAV) electro-optical (EO) gimbal systems are vital for real-time maritime surveillance. However, conventional gimbals often lack edge intelligent perception, and their visual tracking is critically challenged by dynamic fog, causing severe feature collapse and spatio-temporal tracking discontinuities. Existing dehazing-tracking cascades face task misalignment and prohibitive edge latency, while unconstrained domain adaptation is prone to latent manifold collapse. To address these physical and algorithmic limitations, we propose an edge-intelligent UAV EO gimbal system featuring an on-device AI processor. To fully exploit this hardware for onboard multi-vessel tracking, it is empowered by an adaptive vision-language inheritance learning framework, FogTrack++. To maximize algorithm-hardware synergy, a physics-aware fog synthetic engine and a feature-language isomorphic distillation mechanism internalize robust vision-language semantics as offline anchors. Inheriting this degradation-invariant knowledge, a pure-vision lightweight inference pipeline is deployed on the AI processor. Spatially, a language-visual Taylor reparameterization module implicitly models the semantic gradient of the degradation manifold, enabling adaptive feature modulation without complex non-linear overhead. Temporally, a continuous-time stochastic dynamics model absorbs UAV maneuvering jitter into diffusion noise for robust kinematic estimation. Topologically, an information geometry data association strategy elevates tracking instances to statistical probability clouds to combat trajectory fragmentation. Furthermore, a novel high-frame-rate UAV multi-vessel tracking benchmark with paired dynamic fog sequences is introduced. Extensive experiments demonstrate that the tightly coupled system fully exploits the on-device AI processor, achieving state-of-the-art robustness against severe maritime fog while ensuring real-time edge inference.

## Repository Layout

| Path | Description |
| --- | --- |
| `ultralytics/` | Detector code and training scripts |
| `ultralytics/train_foggy.py` | Phase 1/2 training |
| `ultralytics/train_foggy_p3.py` | Phase 3 training |
| `ultralytics/cfg/models/` | Model definitions |
| `ultralytics/weights/` | Detector weights |
| `tracker/` | Tracker code |
| `tracker/track.py` | Tracking implementation |
| `tracker/trackers/` | Tracker implementations |
| `eval/` | TrackEval wrapper |
| `CLIP-main/` | CLIP source |
| `weights/` | Text anchors and ReID weights |
| `track-results/` | Example tracking results |
| `eval_vesselmot_test.py` | Example evaluation script |

## Environment

```bash
conda create -n fogtrackpp python=3.10 -y
conda activate fogtrackpp
pip install -r requirement.txt
```

## Dataset Preparation

### FogMVT++ Dataset

**Download (Baidu Netdisk):** https://pan.baidu.com/s/1xrZoDfcwhDy_twSiewMH5w

### Tracking / Evaluation Data

The default FogMVT++ tracking config is `tracker/config_files/fogmvtpp.yaml`.

```text
datasets/fogmvtpp/inhomogeneous-fog/
+-- test/
    +-- <sequence_name>/
        +-- img1/
        |   +-- 000001.jpg
        |   +-- ...
        +-- gt/
            +-- gt.txt

datasets/fogmvtpp/seqmaps/test-all.txt
```

## Training

Phase 1/2 detector training:

```bash
python ultralytics/train_foggy.py
```

Phase 3 temporal training:

```bash
python ultralytics/train_foggy_p3.py
```

## Tracking

Run FogTrack++ on the FogMVT++ test split:

```bash
/home/user/.conda/envs/fogtrackpp/bin/python tracker/track.py --dataset fogmot --detector Tconv --tracker fogtrack --kalman_format sde
```

## Evaluation

```bash
python eval_vesselmot_test.py
```

The example result summary in `track-results/vessel_summary.txt` reports:

```text
HOTA 50.429
DetA 43.596
AssA 60.829
MOTA 45.281
IDF1 55.182
```

## Contact

If you have any questions, please contact me.

Zijie Zhang

Email: [2410022@tongji.edu.cn](mailto:2410022@tongji.edu.cn)

## Acknowledgements

This project builds on:

- https://github.com/ultralytics/ultralytics
- https://github.com/JackWoo0831/Yolov7-tracker
- https://github.com/openai/CLIP
- https://github.com/YanZhang-zy/BiLaLoRA
- https://github.com/JonathonLuiten/TrackEval

We thank the original authors for their contributions.
