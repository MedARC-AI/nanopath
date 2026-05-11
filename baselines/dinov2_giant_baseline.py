# Run the full frozen-probe suite on untouched Meta DINOv2-G/14-reg weights.
# This reuses the DINOv2 baseline harness so metrics.jsonl matches train.py.

from dinov2_small_baseline import run_dinov2_baseline


if __name__ == "__main__":
    run_dinov2_baseline(
        "dinov2_giant_baseline.py",
        "baseline-dinov2-giant",
        "dinov2-vitg14-reg-no-continued-pretraining",
        "dinov2_vitg14_reg",
        "/data/$USER/nanopath/baselines/dinov2-giant",
        pretrained=True,
    )
