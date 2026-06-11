#!/usr/bin/env python3
# nanopath -> labless bridge. Run from the nanopath repo root after train.py
# finishes; it writes output_dir/labless_submission.json, then posts the same
# payload to labless.

from __future__ import annotations

import datetime as dt
import difflib
import getpass
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


API_URL = "https://api.labless.dev"
PROJECT_SLUG = "nanopath"
NANOPATH_MAIN_REMOTE = "https://github.com/MedARC-AI/nanopath.git"
PRIMARY_METRIC = "mean_probe_score"
LOCKED_PATHS = ("probe.py", "benchmarking/")
FULL_RUN_MIN_FLOPS = 1_000_000_000_000_000_000
FULL_RUN_MAX_SAMPLES = 1_000_000
MAX_REPO_DIFF_BYTES = 120_000
MAX_REVIEW_FILES_BYTES = 500_000
REVIEW_DIFF_PATHS = ("train.py", "model.py", "dataloader.py", "prepare.py")
IGNORED_SOURCE_PATHS = {"AGENTS.md", "CLAUDE.md"}
NANOPATH_LOCKED_PROBE_CONFIG = {
    "enabled": True,
    "model_weights": "ema",
    "count": 1,
    "datasets": ["bracs", "break_his", "mhist", "pcam"],
    "segmentation_datasets": ["pannuke", "monusac", "consep"],
    "slide_datasets": ["ucla_lung"],
    "auc_datasets": ["surgen"],
    "survival_datasets": ["leopard_bcr", "cptac_pda_os"],
    "robustness_datasets": ["pathorob"],
}
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
    api_url = (opts.get("api_url") or API_URL).rstrip("/")
    if truthy(opts.get("login_only", "false")):
        token_path = Path(required(opts, "token_output")).expanduser().resolve()
        github_token, github_login = github_sign_in(api_url)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "github_token": github_token,
            "github_login": github_login,
            "run_name": opts.get("run_name", ""),
            "notes": opts.get("notes", ""),
        }, indent=2) + "\n")
        os.chmod(token_path, 0o600)
        print(f"wrote Labless auto-submit token for {github_login}: {token_path}")
        return 0
    output_dir = Path(required(opts, "output_dir")).expanduser().resolve()
    dry_run = truthy(opts.get("dry_run", "false"))
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
    config_path = public_config_path(opts.get("review_config") or summary.get("config_path") or "configs/main.yaml")
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
    if run_tier == "full" and not validation_errors:
        if number(summary.get("max_train_flops")) != float(FULL_RUN_MIN_FLOPS):
            raise ValueError("full submissions must report max_train_flops=1e18 in summary.json")
        if number(summary.get("max_train_samples")) != float(FULL_RUN_MAX_SAMPLES):
            raise ValueError("full submissions must report max_train_samples=1000000 in summary.json")
        tile_presentations = number(summary.get("tile_presentations"))
        if tile_presentations is None or tile_presentations > FULL_RUN_MAX_SAMPLES:
            raise ValueError("full submissions must not exceed 1000000 tile_presentations")
    run_label = opts.get("run_name") or opts.get("label") or opts.get("title") or run_name
    if run_tier == "full" and len(run_label) > 20:
        raise ValueError("run_name must be 20 characters or fewer")
    if not dry_run and (opts.get("main_commit") or opts.get("main_run_id") or opts.get("source_dir") or opts.get("source_commit") or opts.get("commit")):
        raise ValueError("main/source overrides are only for dry_run=true; real submissions use current GitHub main and output_dir/labless_source")
    repo = collect_source_snapshot(resolve_main(opts, dry_run), summary, opts, output_dir) if run_tier == "full" and not validation_errors else {"locked_path_changes": []}
    validation_errors.extend(f"locked path changed: {p}" for p in repo.pop("locked_path_changes"))
    validation_errors.extend(repo.pop("policy_errors", []))
    env = collect_environment(opts)
    artifacts = collect_artifacts(summary, opts)
    github_token = ""
    github_login = os.environ.get("GITHUB_USER") or getpass.getuser()
    if not dry_run and not validation_errors:
        if opts.get("github_token_file"):
            token_data = json.loads(Path(opts["github_token_file"]).expanduser().read_text())
            github_token = str(token_data["github_token"])
            me_status, me = api_json(api_url, "GET", "/api/auth/github/me", headers={"Authorization": f"Bearer {github_token}"})
            if me_status >= 400:
                raise ValueError(me.get("detail") or f"GitHub identity check failed with HTTP {me_status}")
            github_login = str(me["login"])
            print(f"GitHub signed in as {github_login}", flush=True)
        else:
            github_token, github_login = github_sign_in(api_url)
    baseline_commands = {
        "dinov2-vits14-reg-no-continued-pretraining": "python baselines/dinov2_small_baseline.py configs/main.yaml",
        "dinov2-vitg14-reg-no-continued-pretraining": "python baselines/dinov2_giant_baseline.py configs/main.yaml",
        "genbio-pathfm-vitg16-rope-untouched": "python baselines/genbio_pathfm_baseline.py configs/main.yaml",
    }
    if run_tier == "baseline" and recipe_id not in baseline_commands:
        raise ValueError("baseline is not tracked by labless")
    run_command = opts.get("command") or baseline_commands.get(recipe_id) or f"python train.py {config_path}"
    run_command = public_command(run_command)
    if not opts.get("command") and "output_dir=" not in run_command:
        run_command = f"{run_command} output_dir=$OUTPUT_DIR"
    payload = {
        "version": 1,
        "title": run_label,
        "status": status,
        "notes": opts.get("notes", ""),
        "contributor": {
            "login": github_login,
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

    if dry_run:
        print(json.dumps({"dry_run": True, "status": status, "metric": metric_value, "submission": str(submission_path)}, indent=2))
        return 0

    status_code, result = api_json(
        api_url,
        "POST",
        f"/api/nano-projects/{opts.get('project', PROJECT_SLUG)}/submissions",
        payload,
        {"Authorization": f"Bearer {github_token}"},
    )
    if status_code >= 400:
        raise ValueError(result.get("detail") or f"labless submission failed with HTTP {status_code}")
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


def public_config_path(value: Any) -> str:
    config_path = str(value)
    match = re.search(r"(?:^|/)(configs/[A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml)$", config_path)
    if match:
        return match.group(1)
    if config_path.startswith("/") or "\\" in config_path or not re.match(r"^configs/[A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml$", config_path):
        raise ValueError("summary.config_path must be a repo-relative configs/*.yaml path")
    return config_path


def public_command(value: str) -> str:
    command = value.strip()
    command = re.sub(r"(^|\s)(?:~|/)[^\s\"']*/(configs/[A-Za-z0-9._-]+\.ya?ml)", r"\1\2", command)
    command = re.sub(r"output_dir=['\"](?:~|/)[^'\"]+['\"]", "output_dir=$OUTPUT_DIR", command)
    command = re.sub(r"output_dir=(?:~|/)[^\s\"']+", "output_dir=$OUTPUT_DIR", command)
    return re.sub(r"(^|\s)(?:~|/)[^\s\"']+", r"\1$PATH", command)


def api_json(api_url: str, method: str, path: str, payload: Any = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(api_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("api_url must start with http:// or https://")
    body = json.dumps(payload).encode() if payload is not None else None
    request_headers = {"Accept": "application/json", "User-Agent": "labless-submit/0.1", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.netloc, timeout=30)
    connection.request(method, f"{parsed.path.rstrip('/')}{path}", body=body, headers=request_headers)
    response = connection.getresponse()
    raw = response.read().decode()
    connection.close()
    return response.status, json.loads(raw) if raw else {}


def github_sign_in(api_url: str) -> tuple[str, str]:
    status, device = api_json(api_url, "POST", "/api/auth/github/device", {})
    if status >= 400:
        raise ValueError(device.get("detail") or f"GitHub sign-in failed with HTTP {status}")
    print(f"GitHub sign-in required. Open {device['verification_uri']} and enter code {device['user_code']}.", flush=True)
    deadline = time.monotonic() + int(device["expires_in"])
    interval = int(device["interval"])
    while time.monotonic() < deadline:
        time.sleep(interval)
        status, token = api_json(api_url, "POST", "/api/auth/github/device/token", {"device_code": device["device_code"]})
        if status >= 400:
            raise ValueError(token.get("detail") or f"GitHub sign-in failed with HTTP {status}")
        if token.get("status") == "authorized":
            access_token = str(token["access_token"])
            me_status, me = api_json(api_url, "GET", "/api/auth/github/me", headers={"Authorization": f"Bearer {access_token}"})
            if me_status >= 400:
                raise ValueError(me.get("detail") or f"GitHub identity check failed with HTTP {me_status}")
            print(f"GitHub signed in as {me['login']}", flush=True)
            return access_token, str(me["login"])
        if token.get("status") == "slow_down":
            interval = int(number(token.get("interval")) or interval + 5)
        elif token.get("status") not in {"authorization_pending", "slow_down"}:
            raise ValueError(token.get("error") or "GitHub authorization failed")
    raise ValueError("GitHub sign-in code expired")


def resolve_main(opts: dict[str, str], dry_run: bool) -> dict[str, str]:
    if opts.get("main_commit") or opts.get("main_run_id"):
        if not dry_run:
            raise ValueError("main_run_id/main_commit override is only for dry_run=true")
        main_ref = {"run_id": required(opts, "main_run_id"), "commit": required(opts, "main_commit")}
    else:
        api_url = (opts.get("api_url") or API_URL).rstrip("/")
        project = opts.get("project", PROJECT_SLUG)
        status, main_ref = api_json(api_url, "GET", f"/api/nano-projects/{project}/main")
        if status >= 400:
            raise ValueError(main_ref.get("detail") or f"main lookup failed with HTTP {status}")
        if project == PROJECT_SLUG and api_url == API_URL:
            main_ref["commit"] = current_nanopath_main_commit()
    if not main_ref.get("run_id"):
        raise ValueError("current main response is missing run_id")
    if not isinstance(main_ref.get("commit"), str) or not GIT_SHA_RE.match(main_ref["commit"]):
        raise ValueError("current main response is missing a full 40-character git commit")
    return {"run_id": str(main_ref["run_id"]), "commit": main_ref["commit"]}


def current_nanopath_main_commit() -> str:
    subprocess.run(["git", "fetch", "--depth=1", NANOPATH_MAIN_REMOTE, "refs/heads/main"], check=True)
    commit = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], text=True).strip()
    if not GIT_SHA_RE.match(commit):
        raise ValueError("official nanopath main lookup did not return a full git SHA")
    return commit


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
        "slide_mean_auc": "progression_auc",
        "auc_mean": "mutation_auc",
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


def collect_source_snapshot(main_ref: dict[str, str], summary: dict[str, Any], opts: dict[str, str], output_dir: Path) -> dict[str, Any]:
    meta = summary.get("wandb") if isinstance(summary.get("wandb"), dict) else {}
    git_meta = meta.get("git") if isinstance(meta.get("git"), dict) else {}
    source = str(meta.get("source_artifact") or f"nanopath-source-{meta.get('id', 'local')}")
    local_source_dir = output_dir / "labless_source"
    source_dir = Path(os.path.expandvars(str(opts.get("source_dir") or (local_source_dir if local_source_dir.exists() else meta.get("source_dir") or local_source_dir)))).expanduser().resolve()
    return collect_source_context(main_ref, summary, source_dir, source, opts.get("source_commit") or opts.get("commit") or git_meta.get("commit"), git_meta, opts.get("review_config"))


def collect_source_context(main_ref: dict[str, str], summary: dict[str, Any], source_dir: Path, source: str, commit_value: Any, git_meta: dict[str, Any], review_config: str | None) -> dict[str, Any]:
    if not source_dir.exists():
        raise ValueError(f"source snapshot does not exist: {source_dir}")
    commit = str(commit_value or "")
    if not GIT_SHA_RE.match(commit):
        raise ValueError("source metadata is missing a full 40-character git commit")
    config_rel = public_config_path(review_config or summary.get("config_path") or "configs/main.yaml")
    if not (source_dir / config_rel).exists():
        raise ValueError(f"config snapshot missing: {config_rel}")
    subprocess.run(["git", "cat-file", "-e", f"{main_ref['commit']}^{{commit}}"], check=True)
    review_paths = [*REVIEW_DIFF_PATHS, *([] if config_rel in REVIEW_DIFF_PATHS else [config_rel])]
    main_diff = collect_main_diff(main_ref, commit, source_dir, review_paths)
    review_files = collect_review_files(source, source_dir, review_paths)
    source_changed_files = changed_source_paths(main_ref["commit"], source_dir)
    new_source_files = [path for path in source_changed_files if main_file(main_ref["commit"], path) is None and snapshot_file(source_dir, path) is not None]
    policy_errors = [f"helper file outside allowed surface changed: {path}" for path in source_changed_files if new_source_path_blocked(path)]
    policy_errors.extend(locked_probe_config_errors(source_dir, review_paths))
    repo = {
        "source_artifact": source,
        "review_files": review_files,
        "remote": str(git_meta.get("remote") or ""),
        "branch": "",
        "commit": commit,
        "main_context": main_ref,
        "dirty": bool(main_diff),
        "changed_files": main_diff["files"] if main_diff else [],
        "source_changed_files": source_changed_files,
        "new_source_files": new_source_files,
        "diff_summary": main_diff["summary"] if main_diff else {"files": 0, "added": 0, "removed": 0},
        "locked_path_changes": locked_path_changes(main_ref["commit"], source_dir),
        "policy_errors": policy_errors,
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


def review_path_allowed(path: str) -> bool:
    return path in REVIEW_DIFF_PATHS or bool(re.match(r"^configs/[A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml$", path))


def new_source_path_blocked(path: str) -> bool:
    return not path.startswith("labless/") and Path(path).suffix.lower() in {".py", ".pyi", ".yaml", ".yml"} and not review_path_allowed(path)


def changed_source_paths(commit: str, source_dir: Path) -> list[str]:
    source_files = []
    for p in source_dir.rglob("*"):
        rel_path = p.relative_to(source_dir)
        rel = rel_path.as_posix()
        if p.is_file() and p.name != "manifest.json" and rel not in IGNORED_SOURCE_PATHS and not rel.startswith("labless/") and not any(part.startswith(".") for part in rel_path.parts):
            source_files.append(rel)
    main_files = [
        path for path in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", *REVIEW_DIFF_PATHS, *LOCKED_PATHS], text=True).splitlines()
        if not any(part.startswith(".") for part in Path(path).parts)
    ]
    paths = sorted(set(source_files + main_files))
    return [path for path in paths if main_file(commit, path) != snapshot_file(source_dir, path)]


def locked_probe_config_errors(source_dir: Path, review_paths: list[str]) -> list[str]:
    errors: list[str] = []
    for path in review_paths:
        if not re.match(r"^configs/[A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml$", path):
            continue
        data = snapshot_file(source_dir, path)
        if data is None:
            errors.append(f"locked probe config missing: {path}")
            continue
        config = yaml.safe_load(data.decode("utf-8"))
        probe = config["probe"]
        checked = {key: value for key, value in probe.items() if key != "dataset_roots"}
        if checked != NANOPATH_LOCKED_PROBE_CONFIG:
            errors.append(f"locked probe config changed: {path}")
    return errors


def checked_wandb_url(summary: dict[str, Any], opts: dict[str, str]) -> str:
    url = opts.get("wandb_url") or (summary.get("wandb") if isinstance(summary.get("wandb"), dict) else {}).get("url")
    if url:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.scheme != "https" or parsed.netloc != "wandb.ai" or len(parts) != 4 or parts[2] != "runs":
            raise ValueError("wandb_url must be a W&B run URL like https://wandb.ai/<entity>/<project>/runs/<run_id>")
        return str(url)
    return ""


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
        "patch_bytes": len(patch_bytes),
        "max_patch_bytes": MAX_REPO_DIFF_BYTES,
        "truncated": truncated or bool(omitted),
        "omitted_files": omitted,
    }


def main_file(commit: str, path: str) -> bytes | None:
    exists = subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.check_output(["git", "show", f"{commit}:{path}"]) if exists.returncode == 0 else None


def snapshot_file(source_dir: Path, path: str) -> bytes | None:
    if path in IGNORED_SOURCE_PATHS:
        return None
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
    source_files = []
    for p in source_dir.rglob("*"):
        rel_path = p.relative_to(source_dir)
        rel = rel_path.as_posix()
        if p.is_file() and p.name != "manifest.json" and rel not in IGNORED_SOURCE_PATHS and not rel.startswith("labless/") and not any(part.startswith(".") for part in rel_path.parts):
            source_files.append(rel)
    main_files = [
        path for path in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", *LOCKED_PATHS], text=True).splitlines()
        if not any(part.startswith(".") for part in Path(path).parts)
    ]
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
        "python": sys.version.split()[0],
        "hardware": opts.get("hardware") or gpu or "not reported",
    }


def collect_artifacts(summary: dict[str, Any], opts: dict[str, str]) -> list[dict[str, Any]]:
    wandb_url = checked_wandb_url(summary, opts)
    return [{"kind": "wandb", "uri": wandb_url}] if wandb_url else []


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
