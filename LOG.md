# Experiment log

Running notes on what has been tried in nanopath, with links to wandb where possible. Append new entries at the top. Negative results are valuable! Record them so the next contributor doesn't redo a known dead end.

- _add yours here_

## 2026-04-30: LeJEPA leader replaced by DINOv2 continual pretraining (@tmabraham, @PaulScotti)

This is the run that moved the leaderboard from the old LeJEPA baseline (`0.5228`) to **`0.6373`** by continually pretraining Meta's `dinov2_vits14_reg` on TCGA tiles with DINO CLS + iBOT + KDE. The winning run (`ugwp9367`) used a 1e17-FLOP budget and landed at ~36 min training + ~4 min probes on 1×H100. The current checked-in configs keep the same ViT-S/14 backbone but now target the broader 1e18 cap and the cleaner wall-clock-stop path in `train.py` / `submit/train_1gpu.sbatch`.

[wandb run ugwp9367](https://wandb.ai/paulscotti/nanopath/runs/ugwp9367).

Attribution: the DINOv2 recipe, ablations, and measured lift were Tanishq's contribution. Paul's contribution was the migration into nanopath: reading the full DINOv2-based codebase and rewriting only the important pieces into our current flat files, keeping the smallest practical diff while preserving the parquet data path, probe contract, and no-runtime-DINOv2-dependency style.

Student/teacher pair built from Meta's `dinov2_vits14_reg` checkpoint (ViT-S/14 + 4 register tokens, 22M backbone params), with three loss terms summed:

- **DINO CLS self-distillation** between cross-paired global views (Sinkhorn-Knopp centred teacher targets, 131 072 prototypes, student/teacher temperature `0.1` / cosine `0.04 → 0.07` over the first 27% of FLOPs). Each local view also distills onto each global teacher distribution.
- **iBOT masked-patch self-distillation** with 50% per-image mask probability and 10–45% per-image patch mask ratio.
- **KDE uniformity** on L2-normalised CLS tokens with cross-rank `all_gather`. Off for the first 10% of FLOPs, linearly ramped to full weight by 50%, then constant.

Backbone is trained with AdamW, `lr=1e-4`, layer-wise LR decay `0.7^(11-i)` per block, additional `0.7^12 · 0.2` on the patch_embed, biases/norms with no weight decay, weight decay cosine `0.04 → 0.2`, drop_path `0.1` (linearly per depth), grad clip `3.0`. Teacher backbone + both teacher heads are EMA-updated with momentum cosine `0.994 → 1.0`. Final-layer LR is frozen for the first 0.91% of FLOPs.

Augmentation stack (in `dataloader.py`): HEDJitter `σ=0.05` before crop, RandomResizedCrop, h/v flip, ColorJitter `(0.2, 0.2, 0.2, 0.0)` brightness/contrast/saturation/hue, Normalize. Two global crops at 224 × 224 (scale `[0.32, 1.0]`), eight local crops at 98 × 98 (scale `[0.05, 0.32]`).

Reference points:
- Untouched Meta `dinov2_vits14_reg` (no continual pretraining), measured on Paul's H100 via [`6r1cmaee`](https://wandb.ai/paulscotti/nanopath/runs/6r1cmaee): `0.5838`
- Old LeJEPA `leader_8gpu`: `0.5228`
- This recipe (3-seed-confirmed mean across seeds 1337/2027/4242 in Tanishq's pre-merge sweep): `0.6380` (σ ≈ 0.0007)
- This recipe on Paul's H100 at seed 1337 after the migration: **`0.6373`** ← current leader

Tanishq's pre-merge sweep ran 50+ ablations off a `drop_path=0.1, layerwise_decay=0.7` base on a 4×H100 0.1e18 FLOP budget. Each entry below is a single-knob change off that base; positive deltas survived but did not all clear the 0.01 leaderboard threshold individually. The recipe that ports cleanly to the parquet-shard data path is the cross-rank KDE one — it was the only proven-positive knob that wasn't already on by default and wasn't a slide-reader-side artifact.

**Positive (kept):**
- KDE uniformity (`+0.0035`, +0.0042 with `w=0.003, c=12.5`). Confirmed across three seeds at the production `w=0.002, c=10` setting.
- Stronger color jitter `0.30` (`+0.0022`) — close enough to the default `0.20` that we kept the existing `0.20` to minimise diffs.

**Neutral or negative (rejected):**
- HED jitter applied **after** crop instead of before (`-0.0037`). We keep before-crop.
- Disabling vertical flip (`-0.0057`).
- Teacher temperature `0.10` instead of `0.07` (`-0.0045`).
- ECT-lite `context_size=336` (`-0.0024`).
- Sampling adjacent-tile positives (`-0.0076`).
- AdamW per-parameter grad-RMS clip `0.02` (≈ neutral, `-0.0009`).
- MPP-jitter `[0.75, 1.5]`. The original sweep flagged this as `+0.0041`, but a `read_tile` floor was silently zeroing the zoom-in half of the range — once fixed, symmetric MPP jitter regressed to `+0.0022` at one seed. Irrelevant on parquet anyway since shards are pre-extracted at fixed 224 × 224.
- `color_jitter ≥ 0.4` (`-0.0037`); hue jitter `> 0` (`-0.0033`).
- Mask probability `0.30` (`-0.0033`) or `0.75` (`-0.0009`); default `0.50` is correct.
- `patch_embed_lr_mult` of `0.0` (`-0.0035`) or `0.10` (`-0.0022`); default `0.20` is correct.
- `weight_decay_end=0.10` (≈ flat).
- Muon optimizer on 2D matrices (≈ flat); recipe is AdamW only.
- Anchor loss against the original Meta weights (≈ flat).
- iBOT unmasked-patch distillation weight `> 0` (≈ flat).
- Random grayscale / blur (Tanishq's sweep had them off in the best config; we don't expose them).

Relative to the old LeJEPA leader, the code changed as follows:

- **`model.py`**: replaced `NanoPathFM` (RMSNorm + 2D-RoPE + projector + SIGReg) with a clean ~150-line `DinoV2ViT` that strict-loads Meta's DINOv2 register checkpoints. Attention uses `F.scaled_dot_product_attention` (FlashAttention-2 backend on H100 bf16) — no xformers, no `dinov2.layers` imports. `DINOHead` is reimplemented inline too, so there is no runtime dependency on the dinov2 codebase.
- **`train.py`**: replaced the JEPA + SIGReg loss block with Sinkhorn-Knopp teacher centring + DINO CE + iBOT CE + KDE uniformity. AdamW now uses a single param-group list with per-parameter `lr_mult` (layer-wise decay) and `wd_mult` (zero for biases/norms). EMA covers both heads in addition to the backbone. Validation is now a lightweight held-out DINO/iBOT/KDE diagnostic; leaderboard selection still keys off end-of-run probe score.
- **`probe.py`**: instantiates `DinoV2ViT` (was `NanoPathFM`) and strict-loads the backbone state dict for whatever `cfg["model"]["type"]` requests. Probe contract unchanged: `probe_features(x)` returns the cls token (was: pooled register tokens), `encode_image(x)` returns `[regtokens || patchtokens]` for the seg head.
- **`dataloader.py`**: parquet shard pipeline stayed intact; `v2.ColorJitter` now takes a separate `color_jitter_saturation`, and the dataset exposes both train and held-out patient-hash splits for the lightweight validation pass.
- **`prepare.py`**: added Stage 3 that pulls the Meta DINOv2 weights for the configured `model.type` into the torch hub cache so `train.py` never blocks on the network.
- **`configs/leader.yaml` + `configs/smoke.yaml`**: trimmed the model block to just `type`, dropped the LeJEPA-only `sigreg` / `lambda_*` / projector knobs, added a minimal `dino:` block, and kept lightweight `eval_every` / `val_batches` diagnostics.

Net LOC across the four primary files: 1741 → 1719 (−22 net) despite all the new machinery, because deleting `NanoPathFM` + `SIGReg` + projector + LeJEPA val pass paid for everything new.

## 2026-04-27: **EMA vs. Non-EMA (LeJEPA)**  (@PaulScotti)

Probing the raw model weights instead of EMA showed no difference in performance under LeJEPA. We default to EMA since with OpenMidnight averaging across checkpoints improves results, and EMA approximates that. EMA is also known to enhance robustness. Carried over to the DINOv2 recipe via `probe.model_weights: ema`.
