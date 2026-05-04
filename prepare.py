# Single data-prep entry point. Reads the YAML config the user passes in and
# checks every path train.py will read:
#   - data.dataset_dir/shard-NNNNN.parquet   (the 4M-tile dataset, sharded)
#   - probe.dataset_roots[name] for each of the six probe datasets
#   - Meta's DINOv2 pretrained weights for cfg["model"]["type"] (torch.hub cache)
# Defaults to HF for the tile dataset (medarc/nanopath), each probe's public
# source, and dl.fbaipublicfiles.com for the DINOv2 backbone weights.
# download_TCGA.sh and prepare_tiles / pack_from_jpeg_dir are only relevant if
# you want to regenerate the tile dataset from raw SVS files; see README.
#
# Run:
#   python prepare.py <config.yaml> download=False  # verify only
#   python prepare.py <config.yaml> download=True   # fetch what's missing
#
# `process_row`, `count_rows`, `select_rows`, `prepare_tiles`, and
# `pack_from_jpeg_dir` are kept in this file so a contributor revising tile
# selection can decode a fresh JPEG dataset and pack it into parquet shards
# (see README "Regenerating the tile dataset"); main() does not call them.

import gzip
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import openslide
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image


HF_REPO_ID = "medarc/nanopath"
TILE_SIZE = 224
JPEG_QUALITY = 95
TARGET_TILE_COUNT = 4_000_000
# 200 shards × ~20K JPEGs ≈ ~565 MB/shard at quality 95 — large enough that
# HF transfer is dominated by bytes (not per-file overhead) and small enough
# that a 4 TB shared dataset_dir holds the dataset comfortably.
NUM_SHARDS = 200
# Small row groups inside each parquet shard. The dataloader does random
# per-row reads, and parquet's read_row_group materializes the whole group;
# 64 rows × ~30 KB JPEG ≈ ~2 MB per random access (~2-3 ms incl. decode).
PARQUET_ROW_GROUP_SIZE = 64
# Per-worker LRU; rows are sorted by slide before dispatch so contiguous tiles
# share a handle. Cache=2 covers the boundary when imap_unordered hands a chunk
# from one slide while the previous slide still has tiles in flight.
HANDLE_CACHE_MAX = 2

_HANDLE_CACHE = OrderedDict()
# Suppress repeated logs for a slide we've already marked dead in this worker.
_DEAD_SLIDES = set()


# Open-or-reuse an OpenSlide handle, evicting the LRU and closing it cleanly.
def _get_slide(slide_path):
    slide = _HANDLE_CACHE.get(slide_path)
    if slide is not None:
        _HANDLE_CACHE.move_to_end(slide_path)
        return slide
    while len(_HANDLE_CACHE) >= HANDLE_CACHE_MAX:
        _, old = _HANDLE_CACHE.popitem(last=False)
        old.close()
    slide = openslide.OpenSlide(slide_path)
    _HANDLE_CACHE[slide_path] = slide
    return slide


# Decode one tile and write it as JPEG; returns the manifest-relative path on
# success, None if the slide is unreadable. A poison slide should not kill the
# whole job: log the first failure per slide to stderr and continue. Existing
# files are validated (>0 bytes + JPEG EOF marker) so a partial write left by
# a previous SIGTERM is detected and rewritten. New writes go to a sibling
# ".tmp" file and rename atomically so future runs cannot see partial bytes.
def process_row(args):
    dataset_dir, slide_path, x, y, level = args
    rel = f"{Path(slide_path).stem}/{x}_{y}_{level}.jpg"
    out = Path(dataset_dir) / rel
    if out.exists():
        try:
            with out.open("rb") as f:
                f.seek(-2, os.SEEK_END)
                if f.read(2) == b"\xff\xd9":
                    return rel
        except OSError:
            pass
        out.unlink()
    if slide_path in _DEAD_SLIDES:
        return None
    try:
        slide = _get_slide(slide_path)
        # OpenSlide returns RGBA; drop alpha and emit pure RGB before encoding to JPEG.
        tile = np.asarray(slide.read_region((x, y), level, (TILE_SIZE, TILE_SIZE)))[..., :3]
    except Exception as exc:
        # Drop the broken handle so the next read does not reuse it.
        bad = _HANDLE_CACHE.pop(slide_path, None)
        if bad is not None:
            try:
                bad.close()
            except Exception:
                pass
        if slide_path not in _DEAD_SLIDES:
            print(f"[poison] {slide_path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _DEAD_SLIDES.add(slide_path)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    Image.fromarray(tile).save(tmp, "JPEG", quality=JPEG_QUALITY)
    os.replace(tmp, out)
    return rel


# Count rows in one streaming pass so we never hold all 25M tuples in RAM.
def count_rows(path):
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# Stream-parse only the lines whose 0-indexed row falls in `keep_indices` (sorted).
def select_rows(path, keep_indices):
    keep_iter = iter(keep_indices)
    target = next(keep_iter, None)
    rows = []
    with path.open() as f:
        i = 0
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if target is not None and i == target:
                slide_path, x_str, y_str, level_str = line.rsplit(" ", 3)
                rows.append((slide_path, int(x_str), int(y_str), int(level_str)))
                target = next(keep_iter, None)
            i += 1
            if target is None:
                break
    return rows


# Materialize 4M JPEG tiles from sample_list under dataset_dir. Used to
# regenerate the medarc/nanopath HF mirror when tile selection changes; not
# called by main().
def prepare_tiles(sample_list, dataset_dir, split_seed):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    total = count_rows(sample_list)
    print(f"sample_list rows: {total:,}  ({time.monotonic()-started:.1f}s)", flush=True)
    # Deterministic subsample: same seed across reruns gives the same tile selection.
    if total > TARGET_TILE_COUNT:
        keep = np.random.default_rng(int(split_seed)).choice(total, size=TARGET_TILE_COUNT, replace=False)
        keep.sort()
    else:
        keep = np.arange(total)
    rows = select_rows(sample_list, keep.tolist())
    # Sort by slide so each worker stays on one slide for many consecutive tiles.
    rows.sort(key=lambda r: r[0])
    args_iter = [(str(dataset_dir), *r) for r in rows]
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    print(f"writing {len(args_iter):,} JPEG tiles to {dataset_dir} with {workers} workers", flush=True)
    rels = []
    failed = 0
    decode_started = time.monotonic()
    last_log = decode_started
    with mp.Pool(workers) as pool:
        for i, rel in enumerate(pool.imap_unordered(process_row, args_iter, chunksize=128), start=1):
            if rel is None:
                failed += 1
            else:
                rels.append(rel)
            now = time.monotonic()
            if now - last_log >= 30.0 or i == len(args_iter):
                elapsed = now - decode_started
                rate = i / max(1e-6, elapsed)
                eta = max(0.0, (len(args_iter) - i) / max(1.0, rate))
                print(
                    f"[{i:,}/{len(args_iter):,}]  ok={len(rels):,}  failed={failed:,}  "
                    f"{rate:.0f} tiles/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )
                last_log = now
    manifest_path = dataset_dir / "manifest.txt"
    rels.sort()
    manifest_path.write_text("\n".join(rels) + "\n")
    print(
        f"wrote {manifest_path} with {len(rels):,} entries "
        f"(skipped {failed:,} poison-tile rows; total wall {time.monotonic()-started:.0f}s)",
        flush=True,
    )


# Pack a JPEG-on-disk dataset (the output of prepare_tiles: per-slide subdirs
# + manifest.txt) into NUM_SHARDS parquet shards under out_dir. Step 2 of the
# regen workflow; called by hand after prepare_tiles. File-based to avoid
# materializing 4M JPEG byte-strings (~120 GB) in RAM. Each worker reads the
# JPEGs for its shard chunk and writes one parquet shard with row groups
# sized for cheap random access from the dataloader.
def _pack_one_shard(args):
    jpeg_dir, chunk, out_path = args
    rows = [(p, (jpeg_dir / p).read_bytes()) for p in chunk]
    table = pa.table({"path": [r[0] for r in rows], "jpeg": [r[1] for r in rows]})
    pq.write_table(table, out_path, compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
    return out_path.name, len(chunk), out_path.stat().st_size


def pack_from_jpeg_dir(jpeg_dir, manifest_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(manifest_path.read_text().splitlines())
    chunk_size = (len(paths) + NUM_SHARDS - 1) // NUM_SHARDS
    args_list = [
        (jpeg_dir, paths[i * chunk_size: (i + 1) * chunk_size], out_dir / f"shard-{i:05d}.parquet")
        for i in range(NUM_SHARDS) if paths[i * chunk_size: (i + 1) * chunk_size]
    ]
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    print(f"packing {len(paths):,} tiles into {len(args_list)} parquet shards with {workers} workers", flush=True)
    started = time.monotonic()
    with mp.Pool(workers) as pool:
        for done, (name, n, sz) in enumerate(pool.imap_unordered(_pack_one_shard, args_list), start=1):
            elapsed = time.monotonic() - started
            print(f"[{done}/{len(args_list)}]  {name}: {n:,} rows  {sz/(1<<20):.0f} MB  ({elapsed:.0f}s)", flush=True)


# Pull every shard-NNNNN.parquet from the medarc/nanopath HF dataset into
# dataset_dir. Resumable: huggingface_hub uses a content-addressed cache so
# reruns only fetch what's missing. allow_patterns keeps any non-tile files
# in the repo (README, .gitattributes, etc.) out of dataset_dir.
def fetch_tiles_from_hf(dataset_dir):
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    print(f"downloading parquet shards from huggingface.co/datasets/{HF_REPO_ID} -> {dataset_dir} ({workers} workers)", flush=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        allow_patterns=["shard-*.parquet"],
        max_workers=workers,
    )
    print(f"  [done]  total wall {time.monotonic()-started:.0f}s", flush=True)


# Stream a URL to disk in chunks so large probe archives do not sit in memory.
def http_download(url, dst):
    print(f"  GET {url}\n   -> {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nanopath"})
    with urllib.request.urlopen(req) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def fetch_pannuke(root):
    for fold in (1, 2, 3):
        zip_path = root / f"fold_{fold}.zip"
        http_download(f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()
        (root / f"Fold {fold}").rename(root / f"Fold{fold}")


def fetch_pcam(root):
    base = "https://zenodo.org/api/records/2546921/files"
    for split in ("train", "valid", "test"):
        for kind in ("x", "y"):
            name = f"camelyonpatch_level_2_split_{split}_{kind}.h5"
            gz = root / (name + ".gz")
            http_download(f"{base}/{name}.gz/content", gz)
            with gzip.open(gz, "rb") as fin, (root / name).open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            gz.unlink()


def fetch_bracs(root):
    # BRACS is exposed as FTP, easiest to mirror with wget --recursive.
    cmd = ["wget", "--no-parent", "-nH", "-r", "--directory-prefix", str(root), "ftp://histoimage.na.icar.cnr.it/BRACS_RoI/"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def fetch_break_his(root):
    tar = root / "BreaKHis_v1.tar.gz"
    http_download("http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz", tar)
    shutil.unpack_archive(tar, root)
    tar.unlink()


# MHIST requires manual form access. With download=True we unpack any
# images.zip the user has already dropped into the configured root; otherwise
# we tell them where to put it.
def fetch_mhist(root):
    images_zip = root / "images.zip"
    if images_zip.exists():
        shutil.unpack_archive(images_zip, root)
        images_zip.unlink()
        return
    raise SystemExit(
        f"mhist requires manual access. Visit https://bmirds.github.io/MHIST/#accessing-dataset, "
        f"fill in the form, then drop annotations.csv and images.zip into:\n  {root}\n"
        f"After that, re-run `python prepare.py <config> download=True` to unzip."
    )


# PathoROB ships as two HF datasets (we exclude PathoROB-tcga because TCGA is in our
# pretraining universe). snapshot_download mirrors each subset's parquet under
# pathorob/<subset>/data/ to match the layout probe.py expects.
def fetch_pathorob(root):
    from huggingface_hub import snapshot_download
    for subset in ("camelyon", "tolkach_esca"):
        snapshot_download(repo_id=f"bifold-pathomics/PathoROB-{subset}", repo_type="dataset", local_dir=str(root / subset))


FETCHERS = {
    "bracs": fetch_bracs,
    "break_his": fetch_break_his,
    "mhist": fetch_mhist,
    "pcam": fetch_pcam,
    "pannuke": fetch_pannuke,
    "pathorob": fetch_pathorob,
}


# Resolve $VAR and ~ in a YAML-supplied path string; anything else stays literal.
def _resolve(s):
    return Path(os.path.expanduser(os.path.expandvars(str(s))))


# Flat dict of {label: expanded Path} for every data path declared in cfg.
def get_paths(cfg):
    paths = {"data.dataset_dir": _resolve(cfg["data"]["dataset_dir"])}
    for name, root in cfg["probe"]["dataset_roots"].items():
        paths[f"probe.{name}"] = _resolve(root)
    return paths


# Truthy if the path is populated. mhist needs the unpacked `images/` subdir
# specifically: a user who has dropped only the manual images.zip would
# otherwise look "populated" and we'd skip past fetch_mhist's unpack step.
# pathorob needs both subsets' data/*.parquet (we ignore PathoROB-tcga).
def is_populated(name, p):
    if not p.exists() or not any(p.iterdir()):
        return False
    if name == "mhist" and not (p / "images").exists():
        return False
    if name == "pathorob" and not all(list((p / s / "data").glob("*.parquet")) for s in ("camelyon", "tolkach_esca")):
        return False
    return True


def main():
    usage = "usage: python prepare.py <config.yaml> download=True|download=False"
    # Config path is required, must be a YAML.
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        raise SystemExit(usage)
    config_path = Path(sys.argv[1])
    # download flag is required and must be exactly download=True or download=False.
    if len(sys.argv) != 3 or sys.argv[2] not in ("download=True", "download=False"):
        raise SystemExit(usage)
    download = sys.argv[2] == "download=True"

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    paths = get_paths(cfg)
    dataset_dir = paths["data.dataset_dir"]
    shards = list(dataset_dir.glob("shard-*.parquet")) if dataset_dir.exists() else []

    # Stage 1 — Parquet tile shards (default source: medarc/nanopath HF dataset).
    if shards:
        print(f"[skip] tiles: {dataset_dir} ({len(shards)} shards)", flush=True)
    elif not download:
        raise SystemExit(
            f"no parquet shards (shard-*.parquet) under {dataset_dir}.\n"
            f"Either fix data.dataset_dir in {config_path} to point at an existing prepared "
            f"dataset, or rerun: python prepare.py {config_path} download=True"
        )
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        fetch_tiles_from_hf(dataset_dir)

    # Stage 2 — probe datasets. Verify-only collects every gap and reports
    # them all at once so the user fixes the YAML in a single edit.
    missing = []
    for name in cfg["probe"]["dataset_roots"]:
        root = paths[f"probe.{name}"]
        if is_populated(name, root):
            print(f"[skip] probe/{name}: {root}", flush=True)
            continue
        if not download:
            missing.append((name, root))
            continue
        root.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] probe/{name} -> {root}", flush=True)
        FETCHERS[name](root)
        print(f"[done] probe/{name}", flush=True)

    if missing:
        lines = ["missing probe datasets:"]
        for name, root in missing:
            lines.append(f"  probe/{name}: {root} is empty or missing")
        lines.append(
            f"Either fix probe.dataset_roots in {config_path} to point at existing populated "
            f"paths, or rerun: python prepare.py {config_path} download=True"
        )
        raise SystemExit("\n".join(lines))

    # Stage 3 — Meta's pretrained weights for the model variant in cfg
    # (dinov2_vits14_reg ~84 MB, dinov2_vitb14_reg ~330 MB) live in
    # ~/.cache/torch/hub/checkpoints. model.py:load_dinov2_pretrained streams
    # them on the first forward pass, but pulling them at prep time means
    # train.py never blocks on the network.
    from model import DINOV2_VARIANTS
    import torch
    *_, pretrain_url = DINOV2_VARIANTS[cfg["model"]["type"]]
    weights_dir = Path(torch.hub.get_dir()) / "checkpoints"
    weights_path = weights_dir / Path(pretrain_url).name
    if weights_path.is_file():
        print(f"[skip] dinov2 weights: {weights_path}", flush=True)
    elif not download:
        raise SystemExit(
            f"Meta {cfg['model']['type']} pretrained weights missing at {weights_path}.\n"
            f"Rerun: python prepare.py {config_path} download=True"
        )
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] dinov2 weights -> {weights_path}", flush=True)
        torch.hub.load_state_dict_from_url(pretrain_url, model_dir=str(weights_dir), progress=True)
        print("[done] dinov2 weights", flush=True)

    # Reaching here means tiles + all six probe datasets + DINOv2 weights are
    # in place. Tell the user explicitly so they don't have to read between
    # the [skip] lines.
    n_shards = sum(1 for _ in dataset_dir.glob("shard-*.parquet"))
    print(
        f"\nAll data ready: {n_shards} parquet shards at {dataset_dir}, 6 probe datasets "
        f"({', '.join(cfg['probe']['dataset_roots'])}), and {cfg['model']['type']} weights at "
        f"{weights_path}. Launch training with `python train.py {config_path}` or "
        f"`sbatch submit/train_1gpu.sbatch {config_path}`.",
        flush=True,
    )


if __name__ == "__main__":
    main()
