# Experiment log

Running notes on what has been tried in nanopath, with links to wandb where possible. Append new entries at the top. Negative results are valuable! Record them so the next contributor doesn't redo a known dead end. Make sure to not reference files/logs that are not pushed to main `nanopath` branch.

- _add yours here_

## 2026-05-11: OpenMidnight halfTCGA

`/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` and `/data/OpenMidnight_ckpts/halfTCGA/training_250000/teacher_checkpoint.pth` only differ in that halfTCGA pretrains on half the total data (ignores half the TCGA svs files). The halfTCGA 250k checkpoint does not improve the current 11-probe score, but is very close to the full TCGA OpenMidnight run, suggesting that doubling the pretraining data didn't actually benefit the model much.

| name | mean_probe_score |
|---|---:|
| OpenMidnight default | 0.5499 |
| OpenMidnight halfTCGA 250k | 0.5437 |
