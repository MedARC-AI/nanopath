#!/usr/bin/env python3
# nanopath -> labless bridge. Run from the nanopath repo root after train.py
# finishes; it writes output_dir/labless_submission.json, then posts the same
# payload to labless.

import datetime as dt
import difflib
import getpass
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


API_URL = "https://api.labless.dev"
PROJECT_SLUG = "nanopath-v2"
NANOPATH_MAIN_REMOTE = "https://github.com/MedARC-AI/nanopath.git"
NANOPATH_DEFAULT_BRANCH = "v2"
PRIMARY_METRIC = "final_score"
PROBE_PROTOCOL_VERSION = 2
LOCKED_PATHS = ("probe.py", "benchmarking/")
FULL_RUN_MIN_FLOPS = 1_000_000_000_000_000_000
FULL_RUN_MAX_SAMPLES = 1_000_000
MAX_REPO_DIFF_BYTES = 120_000
MAX_REVIEW_FILES_BYTES = 500_000
MAX_SOURCE_FILE_BYTES = 10_000_000
MAX_GITHUB_TOKEN_FILE_BYTES = 16_384
REVIEW_DIFF_PATHS = ("train.py", "model.py", "dataloader.py", "prepare.py")
IGNORED_SOURCE_PATHS = {"AGENTS.md", "CLAUDE.md"}
NANOPATH_LOCKED_PROBE_CONFIG = {
    "enabled": True,
    "model_weights": "ema",
    "count": 1,
    "datasets": [
        "bach", "bracs", "break_his", "crc", "esca", "mhist", "pcam",
        "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax", "wilds",
    ],
    "segmentation_datasets": ["pannuke", "segpath_epithelial", "segpath_lymphocytes"],
    "slide_datasets": ["ucla_lung"],
    "auc_datasets": ["surgen"],
    "survival_datasets": ["leopard_bcr", "cptac_pda_os"],
    "robustness_datasets": ["pathorob"],
}
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CONFIG_RE = re.compile(r"^configs/[A-Za-z0-9_-][A-Za-z0-9._-]*\.ya?ml$")


def main() -> int:
    os.umask(0o077)
    opts = parse_args(sys.argv[1:])
    api_url = (opts.get("api_url") or API_URL).rstrip("/")
    if str(opts.get("login_only", "false")).strip().lower() in {"1", "true", "yes", "y"}:
        token_path = Path(required(opts, "token_output")).expanduser().absolute()
        github_token, github_login = github_sign_in(api_url)
        write_github_token_file(token_path, {
            "github_token": github_token,
            "github_login": github_login,
            "run_name": opts.get("run_name", ""),
            "notes": opts.get("notes", ""),
        })
        print(f"wrote Labless auto-submit token for {github_login}: {token_path}")
        return 0
    output_dir = Path(required(opts, "output_dir")).expanduser().resolve()
    dry_run = str(opts.get("dry_run", "false")).strip().lower() in {"1", "true", "yes", "y"}
    submission_path = output_dir / "labless_submission.json"
    previous_submission = json.loads(submission_path.read_text()) if submission_path.exists() else {}
    status = opts.get("status", "completed").strip().lower()
    if status != "completed":
        raise ValueError("labless only accepts completed full or baseline nanopath runs")

    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metric_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()] if metrics_path.exists() else []
    json.dumps({"summary": summary, "metrics": metric_rows}, allow_nan=False)
    metric_value = number(summary.get(PRIMARY_METRIC))
    if metric_value is None:
        metric_value = next((value for row in reversed(metric_rows) if (value := number(row.get(PRIMARY_METRIC))) is not None), None)
    validation_errors = validate_output(output_dir, summary_path, metrics_path, summary, metric_rows, metric_value)
    config_path = public_config_path(opts.get("review_config") or summary.get("config_path") or "configs/main.yaml")
    run_name = str(summary.get("project") or output_dir.name)
    recipe_id = str(summary.get("recipe_id") or "")
    run_tier = opts.get("tier") or ("baseline" if summary.get("family") == "baseline" else "full")
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
    if opts.get("command"):
        raise ValueError("command overrides are not supported; Labless derives the public command from the reviewed recipe")
    if not dry_run and (opts.get("main_commit") or opts.get("main_run_id") or opts.get("source_dir") or opts.get("source_commit") or opts.get("commit")):
        raise ValueError("main/source overrides are only for dry_run=true; real submissions use current GitHub main and output_dir/labless_source")
    repo = collect_source_snapshot(resolve_main(opts, dry_run), summary, opts, output_dir) if run_tier == "full" and not validation_errors else {"locked_path_changes": []}
    validation_errors.extend(f"locked path changed: {p}" for p in repo.pop("locked_path_changes"))
    validation_errors.extend(repo.pop("policy_errors", []))
    env = collect_environment(opts)
    wandb_url = checked_wandb_url(summary, opts)
    artifacts = [{"kind": "wandb", "uri": wandb_url}] if wandb_url else []
    github_token = ""
    github_login = os.environ.get("GITHUB_USER") or getpass.getuser()
    if not dry_run and not validation_errors:
        if opts.get("github_token_file"):
            token_data = read_github_token_file(Path(opts["github_token_file"]))
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
        "uni2-h-vith14-untouched": "python baselines/uni2h_baseline.py configs/main.yaml",
        "virchow-vith14-untouched": "python baselines/virchow_baseline.py configs/main.yaml",
    }
    if run_tier == "baseline" and recipe_id not in baseline_commands:
        raise ValueError("baseline is not tracked by labless")
    run_command = baseline_commands.get(recipe_id) or f"python train.py {config_path}"
    if "output_dir=" not in run_command:
        run_command = f"{run_command} output_dir=$OUTPUT_DIR"
    public_summary = {key: value for key, value in summary.items() if key not in {"config_path", "predictive_mean", "mean_probe_score", "final_probe_score", "final_probe_predictive_mean"}}
    public_summary["config_path"] = config_path
    if isinstance(public_summary.get("wandb"), dict):
        public_summary["wandb"] = {key: value for key, value in public_summary["wandb"].items() if key != "source_dir"}
    payload = {
        "version": 1,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
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
            "ended_at": opts.get("ended_at") or previous_submission.get("run", {}).get("ended_at") or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": public_summary,
            "metrics": final_metrics(summary, metric_rows, metric_value),
            "changes": opts.get("changes") or opts.get("notes", ""),
            "environment": env,
            "locked_path_changes": [p.removeprefix("locked path changed: ") for p in validation_errors if p.startswith("locked path changed: ")],
            "validation_errors": validation_errors,
        },
        "artifacts": artifacts,
    }
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


def write_github_token_file(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    if not encoded or len(encoded) > MAX_GITHUB_TOKEN_FILE_BYTES:
        raise ValueError(f"GitHub token file exceeds {MAX_GITHUB_TOKEN_FILE_BYTES} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)


def read_github_token_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().absolute()
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= MAX_GITHUB_TOKEN_FILE_BYTES
    ):
        raise ValueError("GitHub token file must be a current-user mode-0600 regular file with one link")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid, opened.st_nlink, opened.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_nlink,
            metadata.st_size,
        ):
            raise ValueError("GitHub token file changed while it was being opened")
        return json.loads(handle.read())


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
        if status == 404:
            main_ref = {"run_id": "nanopath-v2"}
        elif status >= 400:
            raise ValueError(main_ref.get("detail") or f"main lookup failed with HTTP {status}")
        if (project == PROJECT_SLUG and api_url == API_URL) or not main_ref.get("commit"):
            main_ref["commit"] = current_nanopath_branch_commit()
    if not main_ref.get("run_id"):
        raise ValueError("current main response is missing run_id")
    if not isinstance(main_ref.get("commit"), str) or not GIT_SHA_RE.match(main_ref["commit"]):
        raise ValueError("current main response is missing a full 40-character git commit")
    return {"run_id": str(main_ref["run_id"]), "commit": main_ref["commit"]}


def current_nanopath_branch_commit() -> str:
    subprocess.run(["git", "fetch", "--depth=1", NANOPATH_MAIN_REMOTE, f"refs/heads/{NANOPATH_DEFAULT_BRANCH}"], check=True)
    commit = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], text=True).strip()
    if not GIT_SHA_RE.match(commit):
        raise ValueError(f"official nanopath {NANOPATH_DEFAULT_BRANCH} lookup did not return a full git SHA")
    return commit


def validate_output(output_dir: Path, summary_path: Path, metrics_path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], metric_value: float | None) -> list[str]:
    errors: list[str] = []
    if not output_dir.exists():
        errors.append(f"output_dir does not exist: {output_dir}")
    if not summary_path.exists():
        errors.append("summary.json missing")
    if not metrics_path.exists():
        errors.append("metrics.jsonl missing")
    if metric_value is None:
        errors.append(f"completed run is missing {PRIMARY_METRIC}")
    protocol = number(summary.get("final_probe_protocol_version") or summary.get("probe_protocol_version"))
    if protocol is None:
        protocol = next((number(row.get("probe_protocol_version")) for row in reversed(rows) if number(row.get("probe_protocol_version")) is not None), None)
    if protocol != PROBE_PROTOCOL_VERSION:
        errors.append(f"probe_protocol_version must be {PROBE_PROTOCOL_VERSION}, got {protocol}")
    return errors


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str) and NUMBER_RE.match(value.strip()):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


def final_metrics(summary: dict[str, Any], rows: list[dict[str, Any]], primary: float | None) -> dict[str, float]:
    direct_metrics = {
        "classification_mean_f1", "linear_mean_f1", "knn_mean_f1", "fewshot_mean_f1",
        "seg_mean_f1", "seg_mean_jaccard", "slide_mean_auc", "auc_mean",
        "survival_mean_cindex", "robustness_mean", "robustness_quality_mean",
    }
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        parsed = number(value)
        if key.startswith("final_probe_") and parsed is not None:
            raw = key.removeprefix("final_probe_")
            if raw not in {"predictive_mean", "score"}:
                metrics["probe_protocol_version" if raw == "protocol_version" else raw] = parsed
    for row in rows:
        if row.get("event") == "probe" or row.get("final"):
            for key, value in row.items():
                parsed = number(value)
                if parsed is not None and (key == PRIMARY_METRIC or key.startswith("probe_") or key in direct_metrics):
                    raw = key.removeprefix("probe_")
                    metrics["probe_protocol_version" if raw == "protocol_version" else raw] = parsed
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
    review_files = {"source": source, "files": {path: snapshot_text(source_dir, path) for path in review_paths}}
    if len(json.dumps(review_files, sort_keys=True).encode()) > MAX_REVIEW_FILES_BYTES:
        raise ValueError(f"review files exceed {MAX_REVIEW_FILES_BYTES} bytes")
    source_changed_files = changed_source_paths(commit, source_dir, config_rel)
    new_source_files = [path for path in source_changed_files if main_file(commit, path) is None and snapshot_file(source_dir, path) is not None]
    policy_errors = [f"helper file outside allowed surface changed: {path}" for path in source_changed_files if not path.startswith("labless/") and Path(path).suffix.lower() in {".py", ".pyi", ".yaml", ".yml"} and path not in REVIEW_DIFF_PATHS and not CONFIG_RE.match(path)]
    policy_errors.extend(locked_probe_config_errors(source_dir, review_paths))
    policy_errors.extend(runtime_config_errors(summary, yaml.safe_load(snapshot_file(source_dir, config_rel))))
    repo = {
        "source_artifact": source,
        "review_files": review_files,
        "remote": NANOPATH_MAIN_REMOTE,
        "branch": "",
        "commit": commit,
        "main_context": main_ref,
        "dirty": bool(source_changed_files),
        "changed_files": main_diff["files"] if main_diff else [],
        "source_changed_files": source_changed_files,
        "new_source_files": new_source_files,
        "diff_summary": main_diff["summary"] if main_diff else {"files": 0, "added": 0, "removed": 0},
        "locked_path_changes": [path for path in source_changed_files if path == "probe.py" or path.startswith("benchmarking/")],
        "policy_errors": policy_errors,
    }
    if main_diff:
        repo["main_diff"] = main_diff
    return repo


def changed_source_paths(commit: str, source_dir: Path, config_rel: str) -> list[str]:
    source_files = []
    for p in source_dir.rglob("*"):
        rel_path = p.relative_to(source_dir)
        rel = rel_path.as_posix()
        if CONFIG_RE.match(rel) and rel != config_rel:
            continue
        if p.is_file() and p.name != "manifest.json" and rel not in IGNORED_SOURCE_PATHS and not rel.startswith("labless/") and not any(part.startswith(".") for part in rel_path.parts):
            if (
                p.stat().st_size > MAX_SOURCE_FILE_BYTES
                and p.suffix.lower() not in {".py", ".pyi", ".yaml", ".yml"}
                and not any(rel == lock.rstrip("/") or rel.startswith(lock) for lock in LOCKED_PATHS)
            ):
                continue
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
        if not CONFIG_RE.match(path):
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


# A source snapshot is evidence for the launched run only when its recorded
# tunables agree with the values train.py reported from the in-memory config.
def runtime_config_errors(summary: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors = []
    for section, config_section in config.items():
        if not isinstance(config_section, dict):
            continue
        for config_key, snapshot_value in config_section.items():
            summary_key = "project" if (section, config_key) == ("project", "name") else config_key
            if summary_key not in summary:
                continue
            runtime_value = summary[summary_key]
            runtime_number, snapshot_number = number(runtime_value), number(snapshot_value)
            matches = runtime_number == snapshot_number if runtime_number is not None and snapshot_number is not None else runtime_value == snapshot_value
            if not matches:
                errors.append(
                    f"captured config disagrees with runtime summary: {summary_key}={runtime_value!r:.80}, "
                    f"{section}.{config_key}={snapshot_value!r:.80}"
                )
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
    changed_files, chunks, used, truncated = [], [], 0, False
    summary = {"files": 0, "added": 0, "removed": 0}
    for path in review_paths:
        main_data, source_data = main_file(main_ref["commit"], path), snapshot_file(source_dir, path)
        if main_data == source_data:
            continue
        changed_files.append(path)
        summary["files"] += 1
        patch, file_summary = file_diff(path, main_data, source_data)
        summary["added"] += file_summary["added"]
        summary["removed"] += file_summary["removed"]
        if used < MAX_REPO_DIFF_BYTES:
            encoded = patch.encode()
            room = MAX_REPO_DIFF_BYTES - used
            chunks.append(encoded[:room])
            used += min(len(encoded), room)
            truncated = truncated or len(encoded) > room
        else:
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
        "truncated": truncated,
        "omitted_files": [],
    }


def main_file(commit: str, path: str) -> bytes | None:
    exists = subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.check_output(["git", "show", f"{commit}:{path}"]) if exists.returncode == 0 else None


def snapshot_file(source_dir: Path, path: str) -> bytes | None:
    if path in IGNORED_SOURCE_PATHS:
        return None
    source_path = source_dir / path
    current = source_dir
    for part in Path(path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in source snapshot is forbidden: {path}")
    if not source_path.resolve(strict=False).is_relative_to(source_dir):
        raise ValueError(f"source snapshot path escapes its root: {path}")
    if not source_path.exists() or not source_path.is_file():
        return None
    size = source_path.stat().st_size
    if size > MAX_SOURCE_FILE_BYTES:
        raise ValueError(
            f"large file in source snapshot: {path} ({size} bytes > {MAX_SOURCE_FILE_BYTES}); "
            "keep raw/processed data outside the repo and rerun"
        )
    return source_path.read_bytes()


def snapshot_text(source_dir: Path, path: str) -> str | None:
    data = snapshot_file(source_dir, path)
    if data is None:
        return None
    return data.decode("utf-8")


def file_diff(path: str, main_data: bytes | None, source_data: bytes | None) -> tuple[str, dict[str, int]]:
    old_lines = [] if main_data is None else main_data.decode().splitlines(True)
    new_lines = [] if source_data is None else source_data.decode().splitlines(True)
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
        "python": sys.version.split()[0],
        "hardware": opts.get("hardware") or gpu or "not reported",
    }


if __name__ == "__main__":
    raise SystemExit(main())
