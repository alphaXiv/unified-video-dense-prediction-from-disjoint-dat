# UniD claim reproduction: compact disjoint-task study

[Paper 2607.21592](https://arxiv.org/abs/2607.21592) argues that a pretrained diffusion backbone can absorb specialists trained on disjoint dense-prediction datasets through compact latent projectors, retaining accuracy while avoiding the memory cost of per-pixel distillation. We tested that mechanism with NYU Depth V2 and ADE20K subsets, a 35M-parameter pretrained DDPM U-Net, eight independent long-schedule seeds, matched frozen/direct/pixel controls, and an actual CUDA memory profile.

**Assessment: partially reproduced.** Latent distillation retained **97.3%** of specialist accuracy versus **33.9%** for a frozen backbone, but direct joint training averaged **102.6%** and beat latent on 6/8 seeds. Projector-only fine-tuning raised latent retention to **108.9%**. At eight task decoders, per-pixel training used **4.04 GiB** peak versus **0.54 GiB** for latent training (7.51× total peak; 10.09× incremental) with statistically indistinguishable edge-consistency scores. The paper’s full-scale numbers were 0.059 NYU depth AbsRel, 0.448 ADE20K mIoU after fine-tuning, and an estimated 650/78 GiB pixel/latent profile; our downscaled counterparts were 0.220, 0.023, and 4.04/0.54 GiB.

The reconstruction is intentionally narrower: two image tasks rather than eight video tasks, 64×64 inputs, 512 NYU training examples repartitioned from the labeled validation set, 800 ADE20K training examples for the eight-seed headline, and CIFAR-10 diffusion pretraining rather than Stable Diffusion. The 3,000/5,000-image scale checks are reported separately.

- [Detailed illustrated report](reports/compact-unid/report.md)
- [Self-contained marimo tutorial](reports/compact-unid/notebook.py)
- [Eight-seed measurements](reports/compact-unid/results.csv)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/blob/main/reports/compact-unid/notebook.py)

## Experiment log

All formal runs used Kubernetes. The exact inherited run command was `bash scripts/run_reproduction.sh`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, figures, and polished runnable protocol | Not run as an experiment (publication surface) | Presentation-only | — |
| [Long schedule seeds 0](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-0), [1](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-1), [2](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-2), [3](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-3), [4](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-4), [5](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-5), [6](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-6), [7](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/long-schedule-seed-7) | Fourfold matched schedule; eight independent seeds | `bash scripts/run_reproduction.sh` | Headline: latent 97.3%, direct 102.6%, frozen 33.9%, latent+FT 108.9% retention | 4 GPUs/run; 16 peak; 6.2–9.3 min/run |
| [3k scale](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/scale-3000-ade-longer-train), [5k scale](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/scale-5000-ade-longer-train) | Larger ADE20K subsets and longer schedules | `bash scripts/run_reproduction.sh` | Latent beat direct at 3k, trailed at 5k; projector fine-tuning recovered specialist accuracy | 4 GPUs/run; 4.0/5.2 min |
| [Low-rate FT 120](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/projector-ft-low-lr-120), [240](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/projector-ft-low-lr-240) | Projector/head-only fine-tuning at 3e-5 | `bash scripts/run_reproduction.sh` | 120 steps regressed slightly; 240 steps improved retention by 1.75 points | 4 GPUs/run; 3.0–3.2 min |
| [Random-init control](https://github.com/alphaXiv/unified-video-dense-prediction-from-disjoint-dat/tree/orx/random-init-long-schedule-ablation) | Same architecture/schedule without diffusion-pretrained weights | `bash scripts/run_reproduction.sh` | Worse absolute depth and segmentation for specialist, latent, and direct models | 4 GPUs; 6.6 min |

The complete Kubernetes phase ran for **0.52 elapsed wall-clock hours** on **NVIDIA RTX PRO 6000 Blackwell** GPUs with **16 GPUs peak concurrent**. Failed pre-Python launcher attempts and canceled duplicate branches are retained in the OpenResearch tree but excluded from scientific evidence.

## Run the published protocol

The formal command is:

```bash
bash scripts/run_reproduction.sh
```

It downloads public data and model weights at runtime; no dataset, checkpoint, credential, or restricted artifact is committed. The Kubernetes shape is in `.orx/k8s.yaml`, and scientific settings are in `configs/reproduction.json`.
