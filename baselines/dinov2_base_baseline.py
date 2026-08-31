# Run the full frozen-probe suite on untouched Meta DINOv2-B/14-reg weights.

from dinov2_small_baseline import run_dinov2_baseline


if __name__ == "__main__":
    run_dinov2_baseline(
        "dinov2_base_baseline.py",
        "baseline-dinov2-base",
        "dinov2-vitb14-reg-no-continued-pretraining",
        "dinov2_vitb14_reg",
        "/data/$USER/nanopath/baselines/dinov2-base",
    )
