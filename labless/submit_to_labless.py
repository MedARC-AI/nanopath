#!/usr/bin/env python3
# nanopath -> labless bridge. Run from the nanopath repo root after train.py
# finishes; it writes output_dir/labless_submission.json, then posts the same
# payload to labless.

from __future__ import annotations

import datetime as dt
import difflib
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
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.labless.dev"
PROJECT_SLUG = "nanopath"
PRIMARY_METRIC = "mean_probe_score"
LOCKED_PATHS = ("probe.py", "benchmarking/")
FULL_RUN_MIN_FLOPS = 1_000_000_000_000_000_000
MAX_REPO_DIFF_BYTES = 120_000
MAX_REVIEW_FILES_BYTES = 120_000
REVIEW_DIFF_PATHS = ("train.py", "model.py", "dataloader.py", "prepare.py")
LARGE_DIFF_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".db",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".tar",
    ".webp",
    ".xz",
    ".zip",
)
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    opts = parse_args(sys.argv[1:])
    output_dir = Path(required(opts, "output_dir")).expanduser().resolve()
    submission_path = output_dir / "labless_submission.json"
    previous_submission = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    status = opts.get("status", "completed").strip().lower()
    if status != "completed":
        raise ValueError("labless only accepts completed full or baseline nanopath runs")

    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metric_rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    metric_value = primary_metric(summary, metric_rows)
    validation_errors = validate_output(output_dir, summary_path, metrics_path, metric_value)
    config_path = str(summary.get("config_path") or "configs/leader.yaml")
    run_name = str(summary.get("project") or output_dir.name)
    recipe_id = str(summary.get("recipe_id") or "")
    run_tier = opts.get("tier")
    if not run_tier:
        if summary.get("family") == "baseline":
            run_tier = "baseline"
        else:
            run_tier = "full"
    if run_tier not in {"full", "baseline"}:
        raise ValueError("tier must be full or baseline")
    if run_tier == "full" and "smoke" in config_path:
        raise ValueError("smoke runs are local validation only; submit a completed full run")
    if run_tier == "full" and not validation_errors and number(summary.get("max_train_flops")) != float(FULL_RUN_MIN_FLOPS):
        raise ValueError("full submissions must report max_train_flops=1e18 in summary.json")
    run_label = opts.get("run_name") or opts.get("label") or opts.get("title") or run_name
    if run_tier == "full" and len(run_label) > 20:
        raise ValueError("run_name must be 20 characters or fewer")
    repo = collect_wandb_source(resolve_main(opts), summary, opts) if run_tier == "full" and not validation_errors else {"locked_path_changes": []}
    validation_errors.extend(f"locked path changed: {p}" for p in repo.pop("locked_path_changes"))
    env = collect_environment(opts)
    artifacts = collect_artifacts(output_dir, summary_path, metrics_path, opts)
    baseline_commands = {
        "dinov2-vits14-reg-no-continued-pretraining": "python baselines/dinov2_small_baseline.py configs/leader.yaml",
        "dinov2-vitg14-reg-no-continued-pretraining": "python baselines/dinov2_giant_baseline.py configs/leader.yaml",
        "genbio-pathfm-vitg16-rope-untouched": "python baselines/genbio_pathfm_baseline.py configs/leader.yaml",
    }
    if run_tier == "baseline" and recipe_id not in baseline_commands:
        raise ValueError("baseline is not tracked by labless")
    run_command = opts.get("command") or baseline_commands.get(recipe_id) or f"python train.py {config_path}"
    if not opts.get("command") and "output_dir=" not in run_command:
        run_command = f"{run_command} output_dir={output_dir}"
    payload = {
        "version": 1,
        "title": run_label,
        "status": status,
        "notes": opts.get("notes", ""),
        "contributor": {
            "login": opts.get("contributor") or os.environ.get("GITHUB_USER") or getpass.getuser(),
            "name": opts.get("name") or os.environ.get("GIT_AUTHOR_NAME") or "",
        },
        "repo": repo,
        "run": {
            "name": run_name,
            "label": run_label,
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
            "environment": env,
            "locked_path_changes": [p.removeprefix("locked path changed: ") for p in validation_errors if p.startswith("locked path changed: ")],
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

    if validation_errors:
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


def resolve_main(opts: dict[str, str]) -> dict[str, str]:
    if opts.get("main_commit") or opts.get("main_run_id"):
        main_ref = {"run_id": required(opts, "main_run_id"), "commit": required(opts, "main_commit")}
    else:
        api_url = (opts.get("api_url") or API_URL).rstrip("/")
        project = opts.get("project", PROJECT_SLUG)
        req = urllib.request.Request(
            f"{api_url}/api/nano-projects/{project}/main",
            headers={"Accept": "application/json", "User-Agent": "labless-submit/0.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            main_ref = json.loads(response.read().decode())
    if not main_ref.get("run_id"):
        raise ValueError("current main response is missing run_id")
    if not isinstance(main_ref.get("commit"), str) or not GIT_SHA_RE.match(main_ref["commit"]):
        raise ValueError("current main response is missing a full 40-character git commit")
    return {"run_id": str(main_ref["run_id"]), "commit": main_ref["commit"]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def validate_output(output_dir: Path, summary_path: Path, metrics_path: Path, metric_value: float | None) -> list[str]:
    errors: list[str] = []
    if not output_dir.exists():
        errors.append(f"output_dir does not exist: {output_dir}")
    if not summary_path.exists():
        errors.append("summary.json missing")
    if not metrics_path.exists():
        errors.append("metrics.jsonl missing")
    if metric_value is None:
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


def collect_wandb_source(main_ref: dict[str, str], summary: dict[str, Any], opts: dict[str, str]) -> dict[str, Any]:
    import wandb
    run_path = wandb_run_path(summary, opts)
    api = wandb.Api()
    run = api.run(run_path)
    artifact = api.artifact(f"{run.entity}/{run.project}/nanopath-source-{run.id}:latest", type="code")
    git_meta = run.metadata["git"]
    root = Path.cwd()
    config_path = Path(summary["config_path"])
    config_rel = str(config_path.relative_to(root)) if config_path.is_absolute() else str(config_path)
    subprocess.run(["git", "cat-file", "-e", f"{main_ref['commit']}^{{commit}}"], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = Path(artifact.download(root=tmp))
        review_paths = [*REVIEW_DIFF_PATHS, *([] if config_rel in REVIEW_DIFF_PATHS else [config_rel])]
        main_diff = collect_main_diff(main_ref, git_meta["commit"], source_dir, review_paths)
        review_files = collect_review_files(artifact.qualified_name, source_dir, review_paths)
        repo = {
            "root": str(root),
            "source_artifact": artifact.qualified_name,
            "review_files": review_files,
            "remote": git_meta["remote"],
            "branch": "",
            "commit": git_meta["commit"],
            "main_context": main_ref,
            "dirty": bool(main_diff),
            "changed_files": main_diff["files"] if main_diff else [],
            "diff_summary": main_diff["summary"] if main_diff else {"files": 0, "added": 0, "removed": 0},
            "locked_path_changes": locked_path_changes(main_ref["commit"], source_dir),
        }
        if main_diff:
            repo["main_diff"] = main_diff
        return repo


def collect_review_files(source: str, source_dir: Path, review_paths: list[str]) -> dict[str, Any]:
    files = {path: snapshot_text(source_dir, path) for path in review_paths}
    review_files = {"source": source, "files": files}
    review_bytes = len(json.dumps(review_files, sort_keys=True).encode())
    if review_bytes > MAX_REVIEW_FILES_BYTES:
        raise ValueError(f"review files exceed {MAX_REVIEW_FILES_BYTES} bytes")
    return review_files


def wandb_run_path(summary: dict[str, Any], opts: dict[str, str]) -> str:
    url = opts.get("wandb_url") or (summary.get("wandb") if isinstance(summary.get("wandb"), dict) else {}).get("url")
    if url:
        match = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", url)
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    meta = summary["wandb"]
    return f"{meta['entity']}/{meta['project']}/{meta['id']}"


def collect_main_diff(main_ref: dict[str, str], commit: str, source_dir: Path, review_paths: list[str]) -> dict[str, Any] | None:
    changed_files, omitted, chunks, used, truncated = [], [], [], 0, False
    summary = {"files": 0, "added": 0, "removed": 0}
    for path in review_paths:
        main_data, source_data = main_file(main_ref["commit"], path), snapshot_file(source_dir, path)
        if main_data == source_data:
            continue
        changed_files.append(path)
        summary["files"] += 1
        patch, file_summary, reason = file_diff(path, main_data, source_data)
        summary["added"] += file_summary["added"]
        summary["removed"] += file_summary["removed"]
        if reason:
            omitted.append(reason)
        elif used < MAX_REPO_DIFF_BYTES:
            encoded = patch.encode()
            room = MAX_REPO_DIFF_BYTES - used
            chunks.append(encoded[:room])
            used += min(len(encoded), room)
            truncated = truncated or len(encoded) > room
        else:
            omitted.append(f"{path}: skipped after reaching the {MAX_REPO_DIFF_BYTES} byte patch cap")
            truncated = True
    if not changed_files:
        return None
    patch_bytes = b"".join(chunks)
    return {
        "base_run_id": main_ref["run_id"],
        "base_commit": main_ref["commit"],
        "head_commit": commit,
        "files": changed_files,
        "summary": summary,
        "patch": patch_bytes.decode("utf-8", "replace"),
        "patch_bytes": len(patch_bytes),
        "max_patch_bytes": MAX_REPO_DIFF_BYTES,
        "truncated": truncated or bool(omitted),
        "omitted_files": omitted,
    }


def main_file(commit: str, path: str) -> bytes | None:
    exists = subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.check_output(["git", "show", f"{commit}:{path}"]) if exists.returncode == 0 else None


def snapshot_file(source_dir: Path, path: str) -> bytes | None:
    source_path = source_dir / path
    return source_path.read_bytes() if source_path.exists() and source_path.is_file() else None


def snapshot_text(source_dir: Path, path: str) -> str | None:
    data = snapshot_file(source_dir, path)
    if data is None:
        return None
    if b"\0" in data or path_suffix_is_large(path):
        raise ValueError(f"{path}: review file must be text")
    return data.decode("utf-8")


def file_diff(path: str, main_data: bytes | None, source_data: bytes | None) -> tuple[str, dict[str, int], str]:
    if path_suffix_is_large(path) or (main_data and b"\0" in main_data) or (source_data and b"\0" in source_data):
        return "", {"added": 0, "removed": 0}, f"{path}: skipped binary or large-file patch"
    old_lines = [] if main_data is None else main_data.decode("utf-8", "replace").splitlines(True)
    new_lines = [] if source_data is None else source_data.decode("utf-8", "replace").splitlines(True)
    header = f"diff --git a/{path} b/{path}\n"
    if main_data is None:
        header += f"new file mode 100644\n--- /dev/null\n+++ b/{path}\n"
    elif source_data is None:
        header += f"deleted file mode 100644\n--- a/{path}\n+++ /dev/null\n"
    else:
        header += f"--- a/{path}\n+++ b/{path}\n"
    body = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}")).splitlines(True)[2:]
    patch = header + "".join(body)
    return patch, {
        "added": sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")),
        "removed": sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")),
    }, ""


def locked_path_changes(commit: str, source_dir: Path) -> list[str]:
    source_files = [p.relative_to(source_dir).as_posix() for p in source_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    main_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", *LOCKED_PATHS], text=True).splitlines()
    locked_files = sorted(path for path in set(source_files + main_files) if any(path == lock.rstrip("/") or path.startswith(lock) for lock in LOCKED_PATHS))
    return [path for path in locked_files if main_file(commit, path) != snapshot_file(source_dir, path)]


def path_suffix_is_large(path: str) -> bool:
    return Path(path).suffix.lower() in LARGE_DIFF_SUFFIXES


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
