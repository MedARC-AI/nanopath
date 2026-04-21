import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        raise ValueError("usage: python scripts/rank_runs.py <root> [metric] [descending]")
    root = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) >= 3 else "best_val_mean_probe_f1"
    descending = len(sys.argv) == 4 and sys.argv[3] == "descending"

    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "run": summary_path.parent.name,
                "metric": summary[metric],
                "best_val_lejepa_proxy": summary["best_val_lejepa_proxy"],
                "best_val_mean_probe_f1": summary["best_val_mean_probe_f1"],
                "best_val_mean_probe_f1_step": summary["best_val_mean_probe_f1_step"],
                "model_params": summary["model_params"],
            }
        )
    rows.sort(key=lambda row: (row["metric"] is None, row["metric"]), reverse=descending)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
