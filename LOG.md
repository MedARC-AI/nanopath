# Experiment log

Running notes on what has been tried in nanopath, with links to wandb where possible. Append new entries at the top. Negative results are valuable! Record them so the next contributor doesn't redo a known dead end.

- _add yours here_
- **EMA vs. Non-EMA**: probing the raw model weights instead of EMA shows no difference in performance. I prefer we default to EMA since we know with OpenMidnight that averaging across checkpoints improves results, and EMA is kind of like averaging across checkpoints. Also EMA is known to generally enhance robustness. (@PaulScotti)