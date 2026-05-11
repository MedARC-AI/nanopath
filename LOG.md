# Experiment log

Running notes on what has been tried in nanopath, with links to wandb where possible. Append new entries at the top or let `submit.py` prepend a labless entry. Negative results are valuable! Record them so the next contributor doesn't redo a known dead end.

- _add yours here_

## 2026-05-11: OpenMidnight halfTCGA 250k checkpoint sanity check

`/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` and `/data/OpenMidnight_ckpts/halfTCGA/training_250000/teacher_checkpoint.pth` have the same named `teacher` state layout but are not byte-identical. The halfTCGA 250k checkpoint strict-loads through `load_openmidnight_checkpoint`, but does not improve the current 11-probe score, so the leaderboard keeps the existing OpenMidnight default. Output: `/data/paul/nanopath/baselines/openmidnight-halfTCGA-250000/metrics.jsonl`.

| baseline | 11-probe mean | probe wall |
|---|---:|---:|
| DINOv2-G/14-reg | 0.5615 | 1080.7s |
| OpenMidnight default | 0.5499 | 1189.6s |
| OpenMidnight halfTCGA 250k | 0.5437 | 1087.7s |

## 2026-05-11: DINOv2-G and random DINOv2-S probe baselines

Ran `baselines/dinov2_giant_baseline.py` and `baselines/dinov2_random_baseline.py` on one H100 each with the current 11-probe suite. Outputs: `/data/paul/nanopath/baselines/dinov2-giant/metrics.jsonl` and `/data/paul/nanopath/baselines/dinov2-random/metrics.jsonl`.

| baseline | mean | probe wall |
|---|---:|---:|
| DINOv2-G/14-reg | 0.5615 | 1080.7s |
| random DINOv2-S/14-reg | 0.4283 | 641.3s |

## 2026-05-04: Untouched H-optimus-0 ViT-G archived pre-11-probe baseline

Ran `baselines/hoptimus0_baseline.py` on one H100 with `/data/H-optimus-0/pytorch_model.bin`, H-optimus-0's published normalization stats, and the archived pre-11-probe suite. Output: `/data/paul/nanopath/baselines/hoptimus0-20260504/metrics.jsonl`.

| mean | linear | KNN | few-shot | seg | auc | survival | robustness | probe wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6212 | 0.8031 | 0.7448 | 0.6415 | 0.3015 | 0.6892 | 0.6032 | 0.8791 | 883.0s |

## 2026-05-04: Untouched OpenMidnight ViT-G archived pre-11-probe baseline

Ran `baselines/openmidnight_baseline.py` on one H100 with `/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` and the archived pre-11-probe suite. Output: `/data/paul/nanopath/baselines/openmidnight-20260504/metrics.jsonl`.

| mean | linear | KNN | few-shot | seg | auc | survival | robustness | probe wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5524 | 0.7947 | 0.6759 | 0.4903 | 0.2972 | 0.6020 | 0.4952 | 0.7176 | 885.5s |

## 2026-05-04: Untouched DINOv2-S archived pre-11-probe baseline

Ran `baselines/dinov2_small_baseline.py` on one H100 with untouched Meta `dinov2_vits14_reg` weights and the archived pre-11-probe suite. Output: `/data/paul/nanopath/baselines/dinov2-small-20260504/metrics.jsonl`.

| mean | linear | KNN | few-shot | seg | auc | survival | robustness | probe wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5455 | 0.7249 | 0.6351 | 0.5527 | 0.2541 | 0.7101 | 0.5016 | 0.7466 | 498.3s |
