#!/usr/bin/env python3
# Dependency-free nanopath -> labless bridge. Run from the nanopath repo root
# after train.py finishes; it writes output_dir/labless_submission.json,
# then posts the same payload to labless.

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.labless.dev"
PROJECT_SLUG = "nanopath"
PRIMARY_METRIC = "mean_probe_score"
LOCKED_PATHS = ("probe.py", "benchmarking/")
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def main() -> int:
    opts = parse_args(sys.argv[1:])
    output_dir = Path(required(opts, "output_dir")).expanduser().resolve()
    submission_path = output_dir / "labless_submission.json"
    previous_submission = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    status = opts.get("status", "completed").strip().lower()
    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")

    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metric_rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    metric_value = primary_metric(summary, metric_rows)
    validation_errors = validate_output(output_dir, status, summary_path, metrics_path, metric_value)

    repo = collect_git()
    env = collect_environment(opts)
    artifacts = collect_artifacts(output_dir, summary_path, metrics_path, opts)
    run_name = str(summary.get("project") or output_dir.name)
    config_path = str(summary.get("config_path") or "configs/leader.yaml")
    recipe_id = str(summary.get("recipe_id") or "")
    run_tier = opts.get("tier")
    if not run_tier:
        if summary.get("family") == "baseline":
            run_tier = "baseline"
        elif "smoke" in config_path:
            run_tier = "smoke"
        else:
            run_tier = "full"
    baseline_commands = {
        "dinov2-vits14-reg-no-continued-pretraining": "python baselines/dinov2_small_baseline.py configs/leader.yaml",
        "dinov2-vitg14-reg-no-continued-pretraining": "python baselines/dinov2_giant_baseline.py configs/leader.yaml",
        "dinov2-vits14-reg-random-init-seed0": "python baselines/dinov2_random_baseline.py configs/leader.yaml",
        "genbio-pathfm-vitg16-rope-untouched": "python baselines/genbio_pathfm_baseline.py configs/leader.yaml",
        "hoptimus0-vitg14-reg-untouched": "python baselines/hoptimus0_baseline.py configs/leader.yaml",
        "openmidnight-vitg14-reg-untouched": "python baselines/openmidnight_baseline.py configs/leader.yaml",
    }
    run_command = opts.get("command") or baseline_commands.get(recipe_id) or f"python train.py {config_path}"
    if not opts.get("command") and "output_dir=" not in run_command:
        run_command = f"{run_command} output_dir={output_dir}"
    payload = {
        "version": 1,
        "title": opts.get("title") or f"{summary.get('recipe_id') or run_name} ({repo['branch']})",
        "status": status,
        "notes": opts.get("notes", ""),
        "contributor": {
            "login": opts.get("contributor") or os.environ.get("GITHUB_USER") or getpass.getuser(),
            "name": opts.get("name") or os.environ.get("GIT_AUTHOR_NAME") or "",
        },
        "repo": repo,
        "run": {
            "name": run_name,
            "tier": run_tier,
            "family": summary.get("family") or "nanopath",
            "recipe_id": summary.get("recipe_id"),
            "command": run_command,
            "seed": int(opts["seed"]) if opts.get("seed") else summary.get("config", {}).get("train", {}).get("seed"),
            "hardware": opts.get("hardware") or env["hardware"],
            "started_at": opts.get("started_at"),
            "ended_at": opts.get("ended_at") or previous_submission.get("run", {}).get("ended_at") or now_iso(),
            "summary": summary,
            "metrics": final_metrics(summary, metric_rows),
            "changes": opts.get("changes") or opts.get("notes", ""),
            "failure_reason": opts.get("failure_reason") or (opts.get("notes", "") if status == "failed" else ""),
            "environment": env,
            "locked_path_changes": [p for p in repo["changed_files"] if any(p == lock.rstrip("/") or p.startswith(lock) for lock in LOCKED_PATHS)],
            "validation_errors": validation_errors,
        },
        "artifacts": artifacts,
    }
    if metric_value is not None:
        payload["run"]["metrics"][PRIMARY_METRIC] = metric_value

    payload["submission_id"] = previous_submission.get("submission_id") or hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {submission_path}")

    if validation_errors and status == "completed":
        for error in validation_errors:
            print(f"validation error: {error}", file=sys.stderr)
        return 2

    dry_run = truthy(opts.get("dry_run", "false"))
    if dry_run:
        print(json.dumps({"dry_run": True, "status": status, "metric": metric_value, "submission": str(submission_path)}, indent=2))
        return 0

    req = urllib.request.Request(
        (opts.get("api_url") or API_URL).rstrip("/") + f"/api/nano-projects/{opts.get('project', PROJECT_SLUG)}/submissions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "labless-submit/0.1"},
    )
    if os.environ.get("LABLESS_SUBMIT_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['LABLESS_SUBMIT_TOKEN']}")
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode()
    result = json.loads(body) if body else {"ok": True}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"unsupported argument {arg!r}; use key=value")
        key, value = arg.split("=", 1)
        opts[key.removeprefix("--").replace("-", "_")] = os.path.expandvars(value)
    return opts


def required(opts: dict[str, str], key: str) -> str:
    if not opts.get(key):
        raise ValueError(f"missing required {key}=...")
    return opts[key]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def validate_output(output_dir: Path, status: str, summary_path: Path, metrics_path: Path, metric_value: float | None) -> list[str]:
    errors: list[str] = []
    if not output_dir.exists():
        errors.append(f"output_dir does not exist: {output_dir}")
    if not summary_path.exists():
        errors.append("summary.json missing")
    if not metrics_path.exists():
        errors.append("metrics.jsonl missing")
    if status == "completed" and metric_value is None:
        errors.append(f"completed run is missing {PRIMARY_METRIC} / final_probe_score")
    return errors


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and NUMBER_RE.match(value.strip()):
        return float(value)
    return None


def primary_metric(summary: dict[str, Any], rows: list[dict[str, Any]]) -> float | None:
    for value in (summary.get("final_probe_score"), summary.get(PRIMARY_METRIC), summary.get(f"final_probe_{PRIMARY_METRIC}")):
        parsed = number(value)
        if parsed is not None:
            return parsed
    for row in reversed(rows):
        for key in (PRIMARY_METRIC, "final_probe_score"):
            parsed = number(row.get(key))
            if parsed is not None:
                return parsed
    return None


def final_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    name_map = {
        "score": PRIMARY_METRIC,
        "linear_mean_f1": "linear",
        "knn_mean_f1": "knn",
        "fewshot_mean_f1": "few_shot",
        "seg_mean_jaccard": "seg_jaccard",
        "slide_mean_auc": "slide_auc",
        "auc_mean": "auc",
        "survival_mean_cindex": "survival_cindex",
        "robustness_mean": "robustness",
    }
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        parsed = number(value)
        if key.startswith("final_probe_") and parsed is not None:
            raw = key.removeprefix("final_probe_")
            metrics[name_map.get(raw, raw)] = parsed
    for row in rows:
        if row.get("event") == "probe" or row.get("final"):
            for key, value in row.items():
                parsed = number(value)
                if parsed is not None and (key == PRIMARY_METRIC or key.startswith("probe_") or key in name_map):
                    raw = key.removeprefix("probe_")
                    metrics[name_map.get(raw, raw)] = parsed
    primary = primary_metric(summary, rows)
    if primary is not None:
        metrics[PRIMARY_METRIC] = primary
    return metrics


def collect_git() -> dict[str, Any]:
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
    untracked = [
        line[3:]
        for line in subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
        if line.startswith("?? ")
    ]
    changed_files = sorted(set(changed + staged + untracked))
    return {
        "root": subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip(),
        "remote": subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip(),
        "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(changed_files),
        "changed_files": changed_files,
        "diff_summary": shortstat(subprocess.check_output(["git", "diff", "--shortstat", "HEAD"], text=True).strip()),
    }


def shortstat(text: str) -> dict[str, int]:
    return {
        "files": int(re.search(r"(\d+) files? changed", text).group(1)) if re.search(r"(\d+) files? changed", text) else 0,
        "added": int(re.search(r"(\d+) insertions?", text).group(1)) if re.search(r"(\d+) insertions?", text) else 0,
        "removed": int(re.search(r"(\d+) deletions?", text).group(1)) if re.search(r"(\d+) deletions?", text) else 0,
    }


def collect_environment(opts: dict[str, str]) -> dict[str, Any]:
    gpu = ""
    if shutil.which("nvidia-smi"):
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        if nvidia.returncode == 0:
            gpu = "; ".join(line.strip() for line in nvidia.stdout.splitlines() if line.strip())
    return {
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "hardware": opts.get("hardware") or gpu or f"host:{socket.gethostname()}",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cwd": str(Path.cwd()),
    }


def collect_artifacts(output_dir: Path, summary_path: Path, metrics_path: Path, opts: dict[str, str]) -> list[dict[str, Any]]:
    artifacts = [file_artifact(kind, path) for kind, path in (("summary", summary_path), ("metrics", metrics_path)) if path.exists()]
    artifacts.append({"kind": "submission", "path": str(output_dir / "labless_submission.json")})
    artifacts.extend(file_artifact("slurm_log", path) for path in sorted(Path.cwd().glob("slurm/*.out"), key=lambda p: p.stat().st_mtime)[-3:])
    if opts.get("wandb_url"):
        artifacts.append({"kind": "wandb", "uri": opts["wandb_url"]})
    return artifacts


def file_artifact(kind: str, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"kind": kind, "path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
