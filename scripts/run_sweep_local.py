import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from render_sweep import render_manifest


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        raise ValueError("usage: python scripts/run_sweep_local.py <manifest> [gpu_ids] [python]")

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(sys.argv[1])
    index_path = render_manifest(manifest_path)
    index = json.loads(index_path.read_text())
    variants = index["variants"]
    gpu_ids_arg = sys.argv[2] if len(sys.argv) >= 3 else "0,1,2,3,4,5,6,7"
    python_bin = sys.argv[3] if len(sys.argv) == 4 else str(repo_root / ".venv" / "bin" / "python")
    gpu_ids = [x.strip() for x in gpu_ids_arg.split(",") if x.strip()]
    if len(variants) > len(gpu_ids):
        raise ValueError(f"manifest has {len(variants)} variants but only {len(gpu_ids)} gpu ids were provided")

    log_dir = repo_root / "slurms" / "sweeps" / index["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_id, variant in zip(gpu_ids, variants):
        stdout_path = log_dir / f"{variant['slug']}.out"
        stderr_path = log_dir / f"{variant['slug']}.err"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        stdout = stdout_path.open("w")
        stderr = stderr_path.open("w")
        cmd = [python_bin, "train.py", variant["config_path"]]
        proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=stdout, stderr=stderr)
        processes.append(
            {
                "slug": variant["slug"],
                "gpu_id": gpu_id,
                "proc": proc,
                "stdout": stdout,
                "stderr": stderr,
                "config_path": variant["config_path"],
                "output_dir": variant["output_dir"],
            }
        )
        print(f"launched slug={variant['slug']} gpu={gpu_id} pid={proc.pid} config={variant['config_path']}", flush=True)

    failures = []
    for item in processes:
        code = item["proc"].wait()
        item["stdout"].close()
        item["stderr"].close()
        print(f"finished slug={item['slug']} gpu={item['gpu_id']} code={code}", flush=True)
        if code != 0:
            failures.append((item["slug"], code))
            for other in processes:
                if other["proc"].poll() is None:
                    other["proc"].send_signal(signal.SIGTERM)
            break
    for item in processes:
        if item["proc"].poll() is None:
            item["proc"].wait()
        if not item["stdout"].closed:
            item["stdout"].close()
        if not item["stderr"].closed:
            item["stderr"].close()
    if failures:
        raise SystemExit(f"failed variants: {failures}")


if __name__ == "__main__":
    sys.exit(main())
