# Single data-prep entry point. Reads configs/main.yaml by default (or a
# user-passed YAML config) and checks every path train.py will read:
#   - data.dataset_dir/shard-NNNNN.parquet   (the 4M-tile dataset, sharded)
#   - probe.dataset_roots[name] for each configured probe dataset
#   - pretrained weights for cfg["model"]["type"] (torch.hub cache)
# The tile dataset and protocol-v2 evaluation snapshot are downloaded from their
# separate MedARC Hugging Face repositories when configured roots are missing.
# download_TCGA.sh and prepare_tiles / pack_from_jpeg_dir are only relevant if
# you want to regenerate the tile dataset from raw SVS files; see README.
#
# Run:
#   python prepare.py download=False                    # verify configs/main.yaml
#   python prepare.py download=True                     # fetch what's missing
#   python prepare.py configs/smoke.yaml download=True  # override the default config
#
# `process_row`, `count_rows`, `select_rows`, `prepare_tiles`, and
# `pack_from_jpeg_dir` are kept in this file so a contributor revising tile
# selection can decode a fresh JPEG dataset and pack it into parquet shards
# (see README "Regenerating the tile dataset"); main() does not call them.

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import openslide
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
HF_TRAIN_REPO_ID = "medarc/nanopath"
HF_EVAL_REPO_ID = "medarc/nanopath-evals"
HF_EVAL_REVISION = "cc7e0fdabf622c3877321236dd3c31a0f3cabb15"
TILE_SIZE = 224
JPEG_QUALITY = 95
TARGET_TILE_COUNT = 4_000_000
# 200 shards × ~20K JPEGs ≈ ~565 MB/shard at quality 95 — large enough that
# HF transfer is dominated by bytes (not per-file overhead) and small enough
# that a 4 TB shared dataset_dir holds the dataset comfortably.
NUM_SHARDS = 200
PREPARE_WORKERS = 16
# Small row groups inside each parquet shard. The dataloader does random
# per-row reads, and parquet's read_row_group materializes the whole group;
# 64 rows × ~30 KB JPEG ≈ ~2 MB per random access (~2-3 ms incl. decode).
PARQUET_ROW_GROUP_SIZE = 64
# Per-worker LRU; rows are sorted by slide before dispatch so contiguous tiles
# share a handle. Cache=2 covers the boundary when imap_unordered hands a chunk
# from one slide while the previous slide still has tiles in flight.
HANDLE_CACHE_MAX = 2

_HANDLE_CACHE = OrderedDict()


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


# Decode one tile and write it as JPEG. Existing files are validated (>0 bytes
# plus the JPEG EOF marker), and corrupt inputs fail loudly. New writes go to a
# sibling ".tmp" file and rename atomically so future runs cannot see partial
# bytes.
def process_row(args):
    dataset_dir, slide_path, x, y, level = args
    rel = f"{Path(slide_path).stem}/{x}_{y}_{level}.jpg"
    out = Path(dataset_dir) / rel
    if out.exists() and out.stat().st_size >= 2:
        with out.open("rb") as f:
            f.seek(-2, os.SEEK_END)
            if f.read(2) == b"\xff\xd9":
                return rel
    if out.exists():
        out.unlink()
    slide = _get_slide(slide_path)
    # OpenSlide returns RGBA; drop alpha and emit pure RGB before encoding to JPEG.
    tile = np.asarray(slide.read_region((x, y), level, (TILE_SIZE, TILE_SIZE)))[..., :3]
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
# regenerate the medarc/nanopath training mirror when tile selection changes; not
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
    workers = PREPARE_WORKERS
    print(f"writing {len(args_iter):,} JPEG tiles to {dataset_dir} with {workers} workers", flush=True)
    rels = []
    decode_started = time.monotonic()
    last_log = decode_started
    with mp.Pool(workers) as pool:
        for i, rel in enumerate(pool.imap_unordered(process_row, args_iter, chunksize=128), start=1):
            rels.append(rel)
            now = time.monotonic()
            if now - last_log >= 30.0 or i == len(args_iter):
                elapsed = now - decode_started
                rate = i / max(1e-6, elapsed)
                eta = max(0.0, (len(args_iter) - i) / max(1.0, rate))
                print(
                    f"[{i:,}/{len(args_iter):,}]  "
                    f"{rate:.0f} tiles/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )
                last_log = now
    manifest_path = dataset_dir / "manifest.txt"
    rels.sort()
    manifest_path.write_text("\n".join(rels) + "\n")
    print(
        f"wrote {manifest_path} with {len(rels):,} entries "
        f"(total wall {time.monotonic()-started:.0f}s)",
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
    workers = PREPARE_WORKERS
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
    workers = PREPARE_WORKERS
    print(f"downloading parquet shards from huggingface.co/datasets/{HF_TRAIN_REPO_ID} -> {dataset_dir} ({workers} workers)", flush=True)
    snapshot_download(
        repo_id=HF_TRAIN_REPO_ID,
        repo_type="dataset",
        local_dir=str(dataset_dir),
        allow_patterns=["shard-*.parquet"],
        max_workers=workers,
    )
    print(f"  [done]  total wall {time.monotonic()-started:.0f}s", flush=True)


PATHOBENCH_TILING_VERSION = "pathobench_20x_512_v1"
CPTAC_PDA_OS_TILING_VERSION = PATHOBENCH_TILING_VERSION + "_cptac_pda_os_fold0_train_v1"
UCLA_LUNG_TILING_VERSION = PATHOBENCH_TILING_VERSION
SURGEN_TILING_VERSION = PATHOBENCH_TILING_VERSION
LEOPARD_BCR_TILING_VERSION = PATHOBENCH_TILING_VERSION + "_leopard_bcr_174x768_v1"


def fetch_eval_dataset(name, root):
    from huggingface_hub import snapshot_download
    import tarfile
    import zipfile

    workers = PREPARE_WORKERS
    download_dir = root.parent / f".nanopath-evals-{name}"
    patterns = ["manifest.json", f"archives/{name}.tar"]
    if name == "pannuke":
        patterns = ["manifest.json", "archives/pannuke/*.zip"]
    elif name in {"ucla_lung", "surgen", "leopard_bcr", "cptac_pda_os", "pathorob"}:
        patterns = ["manifest.json", f"datasets/{name}/**"]
    print(f"  downloading {name} from huggingface.co/datasets/{HF_EVAL_REPO_ID}@{HF_EVAL_REVISION}", flush=True)
    snapshot_download(
        repo_id=HF_EVAL_REPO_ID,
        repo_type="dataset",
        revision=HF_EVAL_REVISION,
        local_dir=str(download_dir),
        allow_patterns=patterns,
        max_workers=workers,
    )
    release = json.loads((download_dir / "manifest.json").read_text())
    assert release["probe_protocol_version"] == 2
    assert release["contains_official_test_records"] is False
    for manifest_name, expected_sha in release["source_manifests_sha256"].items():
        assert hashlib.sha256((REPO_ROOT / "benchmarking" / manifest_name).read_bytes()).hexdigest() == expected_sha
    release_paths = {item["path"] for item in release["files"]}
    if name == "pannuke":
        assert {"archives/pannuke/fold_1.zip", "archives/pannuke/fold_2.zip"} <= release_paths
    elif name in {"ucla_lung", "surgen", "leopard_bcr", "cptac_pda_os", "pathorob"}:
        assert any(path.startswith(f"datasets/{name}/") for path in release_paths)
    else:
        assert f"archives/{name}.tar" in release_paths
    root.mkdir(parents=True, exist_ok=True)
    if name == "pannuke":
        for archive in sorted((download_dir / "archives" / "pannuke").glob("*.zip")):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(root)
    elif (archive := download_dir / "archives" / f"{name}.tar").is_file():
        with tarfile.open(archive) as tf:
            tf.extractall(root, filter="data")
    else:
        source = download_dir / "datasets" / name
        for path in sorted(source.rglob("*")):
            if path.is_file():
                destination = root / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    shutil.rmtree(download_dir)


# Resolve $VAR and ~ in a YAML-supplied path string; anything else stays literal.
def _resolve(s):
    return Path(os.path.expanduser(os.path.expandvars(str(s))))


# Keep populated shared roots and writable /data defaults unchanged. On machines
# without those mounts, retarget missing data into the clone's ignored data/ dir.
def _local_data_root(s):
    p = _resolve(s)
    if p.is_dir() and any(p.iterdir()):
        return str(s)
    if p.is_absolute() and len(p.parts) > 1 and p.parts[1] in {"data", "block"} and Path(*p.parts[:2]).exists() and os.access(Path(*p.parts[:2]), os.W_OK):
        return str(s)
    if p.is_absolute() and len(p.parts) > 1 and p.parts[1] in {"data", "block"}:
        return str(REPO_ROOT / "data" / p.name)
    return str(s)


# Output/log roots follow repo-local data roots when prepare had to localize a
# config, otherwise they stay on writable shared /data for MedARC runs.
def _local_output_root(s, force=False):
    p = _resolve(s)
    mount = Path(*p.parts[:2]) if p.is_absolute() and len(p.parts) > 1 else p
    if force or (p.is_absolute() and not p.exists() and (not mount.exists() or not os.access(mount, os.W_OK))):
        parts = list(p.parts)
        tail = parts[parts.index("nanopath") + 1:] if "nanopath" in parts else [p.name]
        return str(REPO_ROOT / "data" / Path(*tail))
    return str(s)


# Rewrite missing portable defaults in place before downloading. Surgical text
# replacement preserves comments and formatting, so the YAML remains the source
# of truth that train.py/probe.py read unchanged after preparation.
def localize_config_file(config_path):
    raw = config_path.read_text()
    cfg = yaml.safe_load(raw)
    data_roots = [cfg["data"]["dataset_dir"], *cfg["probe"]["dataset_roots"].values()]
    output_roots = [cfg["project"]["output_dir"], cfg["project"]["wandb_dir"]]
    changes = {v: nv for v in data_roots if (nv := _local_data_root(v)) != v}
    changes.update({v: nv for v in output_roots if (nv := _local_output_root(v, force=bool(changes))) != v})
    for old, new in changes.items():
        raw = raw.replace(f": {old}", f": {new}")
    if changes:
        config_path.write_text(raw)
        print(f"[data] rewrote {len(changes)} missing/unusable root(s) in {config_path} to defaults under {REPO_ROOT / 'data'}.", flush=True)


def localize_config_files(config_path):
    # Smoke is the usual first command on a fresh clone, but users naturally
    # train main next. Keep both checked-in recipes pointed at the same local
    # downloaded data once either config triggers localization.
    seen = set()
    for path in [config_path, REPO_ROOT / "configs" / "main.yaml", REPO_ROOT / "configs" / "smoke.yaml"]:
        path = path.resolve()
        if path.exists() and path not in seen:
            seen.add(path)
            localize_config_file(path)


# Flat dict of {label: expanded Path} for every data path declared in cfg.
def get_paths(cfg):
    paths = {"data.dataset_dir": _resolve(cfg["data"]["dataset_dir"])}
    for name, root in cfg["probe"]["dataset_roots"].items():
        paths[f"probe.{name}"] = _resolve(root)
    return paths


# Truthy if the path is populated with files train.py/probe.py actually read,
# not merely a half-written archive left by an interrupted download.
def is_populated(name, p):
    if not p.exists():
        return False
    bench = Path(__file__).resolve().parent / "benchmarking"
    thunder = json.loads((bench / "thunder_v2.json").read_text())
    assert set(thunder) == {"protocol_version", "seed", "classification", "segmentation"}
    assert thunder["protocol_version"] == 2
    assert thunder["seed"] == 1337
    assert all(set(spec) == {"root", "train", "val"} for family in ("classification", "segmentation") for spec in thunder[family].values())
    if name in thunder["classification"]:
        spec = thunder["classification"][name]
        train_counts = np.bincount(np.asarray(spec["train"]["labels"], dtype=np.int64))
        val_counts = np.bincount(np.asarray(spec["val"]["labels"], dtype=np.int64), minlength=len(train_counts))
        if len(train_counts) == 0 or train_counts.min() < 16 or val_counts.min() == 0:
            return False
        if name == "pcam":
            import h5py
            if len(spec["train"]["indices"]) != len(spec["train"]["labels"]) or len(spec["val"]["indices"]) != len(spec["val"]["labels"]):
                return False
            for split, source_split in (("train", "train"), ("val", "valid")):
                x_path = p / f"camelyonpatch_level_2_split_{source_split}_x.h5"
                y_path = p / f"camelyonpatch_level_2_split_{source_split}_y.h5"
                if not x_path.is_file() or not y_path.is_file():
                    return False
                indices = np.asarray(spec[split]["indices"], dtype=np.int64)
                with h5py.File(x_path, "r") as x, h5py.File(y_path, "r") as y:
                    x_values, y_values = x[next(iter(x))], y[next(iter(y))]
                    if indices[-1] >= len(x_values) or indices[-1] >= len(y_values):
                        return False
                    if not np.array_equal(np.asarray(y_values[indices]).reshape(-1), np.asarray(spec[split]["labels"])):
                        return False
            return True
        if any(len(spec[split]["images"]) != len(spec[split]["labels"]) for split in ("train", "val")):
            return False
        return not (set(spec["train"]["images"]) & set(spec["val"]["images"])) and all((p / rel).is_file() for split in ("train", "val") for rel in spec[split]["images"])
    if name in thunder["segmentation"]:
        spec = thunder["segmentation"][name]
        if name == "pannuke":
            paths = [spec[split][kind] for split in ("train", "val") for kind in ("images", "labels")]
            if not (
                all(set(spec[split]) == {"images", "labels"} for split in ("train", "val"))
                and not any("fold3" in path.lower() or "test" in path.lower() for path in paths)
                and set(spec["train"].values()).isdisjoint(spec["val"].values())
                and all((p / path).is_file() for path in paths)
            ):
                return False
            train_images = np.load(p / spec["train"]["images"], mmap_mode="r")
            train_masks = np.load(p / spec["train"]["labels"], mmap_mode="r")
            val_images = np.load(p / spec["val"]["images"], mmap_mode="r")
            val_masks = np.load(p / spec["val"]["labels"], mmap_mode="r")
            return (
                train_images.shape == (2656, 256, 256, 3)
                and train_masks.shape == (2656, 256, 256, 6)
                and val_images.shape == (2523, 256, 256, 3)
                and val_masks.shape == (2523, 256, 256, 6)
            )
        train_images, val_images = set(map(tuple, spec["train"]["images"])), set(map(tuple, spec["val"]["images"]))
        train_labels, val_labels = set(map(tuple, spec["train"]["labels"])), set(map(tuple, spec["val"]["labels"]))
        records = [record for split in ("train", "val") for kind in ("images", "labels") for record in spec[split][kind]]
        train_sources = {record[0] for record in spec["train"]["images"]}
        val_sources = {record[0] for record in spec["val"]["images"]}
        paired = all(len(spec[split]["images"]) == len(spec[split]["labels"]) for split in ("train", "val"))
        return paired and not (train_images & val_images) and not (train_labels & val_labels) and not (train_sources & val_sources) and all((p / record[0]).is_file() for record in records)
    if name == "ucla_lung":
        splits = json.loads((bench / "ucla_lung.json").read_text())
        expected = set(splits["train"]["slide_ids"] + splits["val"]["slide_ids"])
        got = set(pq.read_table(p / "tiles.parquet", columns=["slide_id"]).column("slide_id").to_pylist()) if (p / "tiles.parquet").exists() else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == UCLA_LUNG_TILING_VERSION and expected <= got
    if name == "surgen":
        splits = json.loads((bench / "surgen.json").read_text())
        expected = set(splits["train"]["slides"] + splits["val"]["slides"])
        files = sorted((p / "data").glob("surgen-*.parquet"))
        labels = {line.split(",")[0] for line in (p / "labels.csv").read_text().splitlines()[1:]} if (p / "labels.csv").exists() else set()
        got = set(pa.concat_tables([pq.read_table(f, columns=["slide_id"]) for f in files]).column("slide_id").to_pylist()) if files else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == SURGEN_TILING_VERSION and expected <= labels and expected <= got
    if name == "leopard_bcr":
        splits = json.loads((bench / "leopard_bcr.json").read_text())
        expected = {sid for slides in splits["case_slides"] for sid in slides}
        got = set(pq.read_table(p / "patches.parquet", columns=["slide_id"]).column("slide_id").to_pylist()) if (p / "patches.parquet").exists() else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == LEOPARD_BCR_TILING_VERSION and (p / "labels.tsv").exists() and expected <= got
    if name == "cptac_pda_os":
        splits = json.loads((bench / "cptac_pda_os.json").read_text())
        expected = {sid for slides in splits["case_slides"] for sid in slides}
        got = set(pq.read_table(p / "patches.parquet", columns=["slide_id"]).column("slide_id").to_pylist()) if (p / "patches.parquet").exists() else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == CPTAC_PDA_OS_TILING_VERSION and (p / "labels.tsv").exists() and expected <= got
    if name == "pathorob":
        return all(list((p / subset / "data").glob("*.parquet")) for subset in ("camelyon", "tolkach_esca"))
    return True


def main():
    usage = "usage: python prepare.py [config.yaml] download=True|download=False"
    args = sys.argv[1:]
    # The download flag is required and must be exactly download=True or download=False.
    if not args or args[-1] not in ("download=True", "download=False"):
        raise SystemExit(usage)
    download = args[-1] == "download=True"
    # Config path is optional; without one, prepare the canonical main recipe.
    if len(args) == 1:
        config_path = REPO_ROOT / "configs" / "main.yaml"
    elif len(args) == 2 and args[0].endswith((".yaml", ".yml")):
        config_path = Path(args[0])
    else:
        raise SystemExit(usage)
    resolved_config_path = config_path.resolve()
    config_label = str(resolved_config_path.relative_to(REPO_ROOT)) if resolved_config_path.is_relative_to(REPO_ROOT) else str(config_path)
    prepare_cmd = "python prepare.py download=True" if len(args) == 1 else f"python prepare.py {config_label} download=True"

    # Off-cluster, correct the requested config plus the checked-in smoke/main
    # recipes before preparing, so subsequent train.py commands read the same
    # local paths that download=True populates.
    if download:
        localize_config_files(config_path)
    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    paths = get_paths(cfg)
    dataset_dir = paths["data.dataset_dir"]
    shards = list(dataset_dir.glob("shard-*.parquet")) if dataset_dir.exists() else []

    # Stage 1 — Parquet tile shards (default source: medarc/nanopath HF dataset).
    if len(shards) == NUM_SHARDS:
        print(f"[verify] tiles: {dataset_dir} ({len(shards)} shards)", flush=True)
    elif not download:
        raise SystemExit(
            f"expected {NUM_SHARDS} parquet shards under {dataset_dir}, found {len(shards)}.\n"
            f"Either fix data.dataset_dir in {config_label} to point at an existing prepared "
            f"dataset, or rerun: {prepare_cmd}"
        )
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        fetch_tiles_from_hf(dataset_dir)
        assert sum(1 for _ in dataset_dir.glob("shard-*.parquet")) == NUM_SHARDS, f"tiles still incomplete after fetch: {dataset_dir}"

    # Stage 2 — probe datasets. Verify-only collects every gap and reports
    # them all at once so the user fixes the YAML in a single edit.
    missing = []
    for name in cfg["probe"]["dataset_roots"]:
        root = paths[f"probe.{name}"]
        if is_populated(name, root):
            print(f"[verify] probe/{name}: {root}", flush=True)
            continue
        if not download:
            missing.append((name, root))
            continue
        root.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] probe/{name} -> {root}", flush=True)
        fetch_eval_dataset(name, root)
        assert is_populated(name, root), f"probe/{name} is still missing, empty, or stale after fetch: {root}"
        print(f"[done] probe/{name}", flush=True)

    if missing:
        lines = ["missing probe datasets:"]
        for name, root in missing:
            lines.append(f"  probe/{name}: {root} is empty, missing, or stale for the current benchmark")
        lines.append(
            f"Either fix probe.dataset_roots in {config_label} to point at existing populated "
            f"paths, or rerun: {prepare_cmd}"
        )
        raise SystemExit("\n".join(lines))

    # Stage 3 — pretrained weights for the model variant in cfg
    # (small ~84 MB, base ~330 MB, large ~1.2 GB, giant ~4 GB) live in
    # ~/.cache/torch/hub/checkpoints. Pulling them at prep time means train.py
    # never blocks on the network.
    from model import VIT_VARIANTS
    import torch
    pretrain_url = VIT_VARIANTS[cfg["model"]["type"]][7]
    weights_dir = Path(torch.hub.get_dir()) / "checkpoints"
    weights_path = weights_dir / Path(pretrain_url).name
    if weights_path.is_file():
        print(f"[skip] model weights: {weights_path}", flush=True)
    elif not download:
        raise SystemExit(
            f"{cfg['model']['type']} pretrained weights missing at {weights_path}.\n"
            f"Rerun: {prepare_cmd}"
        )
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] model weights -> {weights_path}", flush=True)
        torch.hub.load_state_dict_from_url(pretrain_url, model_dir=str(weights_dir), progress=True)
        print("[done] model weights", flush=True)

    # Reaching here means tiles + every configured probe dataset + model weights are
    # in place. Tell the user explicitly so they don't have to read between
    # the [skip] lines.
    n_shards = sum(1 for _ in dataset_dir.glob("shard-*.parquet"))
    n_probes = len(cfg["probe"]["dataset_roots"])
    print(
        f"\nAll data ready: {n_shards} parquet shards at {dataset_dir}, {n_probes} probe datasets "
        f"({', '.join(cfg['probe']['dataset_roots'])}), and {cfg['model']['type']} weights at "
        f"{weights_path}. Launch training with `python train.py {config_label}` or "
        f"`./submit/train_1gpu.sbatch {config_label}`.",
        flush=True,
    )


if __name__ == "__main__":
    main()
