# Run the full frozen-probe suite on untouched Meta DINOv2-L/14-reg weights.
# This reuses the DINOv2 baseline harness so metrics.jsonl matches train.py.

from dinov2_small_baseline import run_dinov2_baseline


if __name__ == "__main__":
    run_dinov2_baseline(
        "dinov2_large_baseline.py", "baseline-dinov2-large",
        "dinov2-vitl14-reg-no-continued-pretraining", "dinov2_vitl14_reg",
        "/data/$USER/nanopath/baselines/dinov2-large",
    )
