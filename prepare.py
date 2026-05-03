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
import io
import json
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


def fetch_bach(root):
    base = "https://zenodo.org/api/records/3632035/files"
    for name in ("ICIAR2018_BACH_Challenge.zip", "ICIAR2018_BACH_Challenge_TestDataset.zip"):
        zip_path = root / name
        http_download(f"{base}/{name}/content", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()


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


# PathoROB (Kömen et al. 2025) provides each dataset as parquet shards on the HF Hub.
# We pull camelyon + tolkach_esca only — the benchmark's TCGA cohort overlaps with our
# pretraining tile universe, so it isn't held-out and is excluded from our robustness eval.
def fetch_pathorob(root):
    from huggingface_hub import snapshot_download
    for name in ("camelyon", "tolkach_esca"):
        snapshot_download(
            repo_id=f"bifold-pathomics/PathoROB-{name}",
            repo_type="dataset",
            local_dir=str(root / name),
            allow_patterns=["data/*.parquet"],
        )


# MoNuSAC's official release ships split zips behind a Google Drive form; there's no clean
# direct-download URL. We mirror the Thunder convention: the user drops the two release zips
# (`MoNuSAC_images_and_annotations.zip` + `MoNuSAC Testing Data and Annotations.zip`) into the
# configured root and we just unpack them on download=True.
def fetch_monusac(root):
    train_zip = root / "MoNuSAC_images_and_annotations.zip"
    test_zip = root / "MoNuSAC Testing Data and Annotations.zip"
    if train_zip.exists() and test_zip.exists():
        for z in (train_zip, test_zip):
            shutil.unpack_archive(z, root)
            z.unlink()
        return
    raise SystemExit(
        f"monusac requires manual access from https://monusac-2020.grand-challenge.org/Data/. "
        f"Drop `MoNuSAC_images_and_annotations.zip` and `MoNuSAC Testing Data and Annotations.zip` "
        f"into:\n  {root}\nThen re-run `python prepare.py <config> download=True` to unpack."
    )


# Chimera (CHIMERA bladder cancer challenge, https://chimera.grand-challenge.org/) task3:
# 126 train (3A) + 50 held-out (3B) NMIBC WSIs with `progression` binary labels in *_CD.json
# clinical metadata. Slides are huge .tif (~2 GB each) so probing slide-level cannot decode
# them on the fly — we pre-extract a fixed sample of ~256 random tissue tiles per slide
# (cached as parquet shards under /data/chimera/chimera_tiles), and probe.py mean-pools the
# embedded tiles per slide to score AUROC. fetch_chimera() just routes the user to the
# separate prepare_chimera.sbatch (download + extract is ~1.5 hr, too slow for prepare.py).
CHIMERA_BUCKET = "s3://chimera-challenge/v2/task3/data"
CHIMERA_LEVEL0_TILE = 448  # 224 px × 2 to read 0.48 mpp from a 0.24 mpp level-0 pyramid.
CHIMERA_OUT_TILE = 224
CHIMERA_TILES_PER_SLIDE = 256
CHIMERA_RNG_SEED = 1337


def fetch_chimera_raw(raw_dir):
    """Sync HE.tif + HE_mask.tif for every 3A/3B chimera task3 case from the public S3 bucket."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync", CHIMERA_BUCKET, str(raw_dir),
        "--no-sign-request",
        "--exclude", "*",
        "--include", "3A_*/*_HE.tif", "--include", "3A_*/*_HE_mask.tif",
        "--include", "3B_*/*_HE.tif", "--include", "3B_*/*_HE_mask.tif",
        "--include", "3A_*/*_CD.json", "--include", "3B_*/*_CD.json",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# Read one slide + its tissue mask, pick CHIMERA_TILES_PER_SLIDE random foreground centers,
# crop a 448x448 level-0 region around each, resize to 224x224 (~0.48 mpp), JPEG-encode.
def _chimera_extract_one(case_dir):
    case_id = case_dir.name
    cohort = case_id[:2]
    cd = json.loads((case_dir / f"{case_id}_CD.json").read_text())
    progression = int(cd["progression"])
    import tifffile
    mask = tifffile.imread(case_dir / f"{case_id}_HE_mask.tif")
    slide = openslide.OpenSlide(str(case_dir / f"{case_id}_HE.tif"))
    w0, h0 = slide.dimensions
    scale = w0 / mask.shape[1]  # mask px → level-0 px (mask aligns with a pyramid level)
    fg = np.argwhere(mask > 0)
    rng = np.random.default_rng(CHIMERA_RNG_SEED + int(case_id.split("_")[1]))
    n = min(CHIMERA_TILES_PER_SLIDE, len(fg))
    chosen = rng.choice(len(fg), size=n, replace=False)
    rows = []
    for i in chosen:
        my, mx = fg[i]
        x0 = max(0, min(w0 - CHIMERA_LEVEL0_TILE, int(mx * scale) - CHIMERA_LEVEL0_TILE // 2))
        y0 = max(0, min(h0 - CHIMERA_LEVEL0_TILE, int(my * scale) - CHIMERA_LEVEL0_TILE // 2))
        tile = np.asarray(slide.read_region((x0, y0), 0, (CHIMERA_LEVEL0_TILE, CHIMERA_LEVEL0_TILE)))[..., :3]
        buf = io.BytesIO()
        Image.fromarray(tile).resize((CHIMERA_OUT_TILE, CHIMERA_OUT_TILE), Image.BILINEAR).save(buf, "JPEG", quality=95)
        rows.append((buf.getvalue(), case_id, cohort, progression))
    slide.close()
    return case_id, rows


def prepare_chimera_tiles(raw_dir, tile_dir):
    """Walk every 3A/3B case under raw_dir, extract tiles, write one parquet shard + labels.csv."""
    out_data = tile_dir / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and (d / f"{d.name}_HE.tif").exists() and (d / f"{d.name}_HE_mask.tif").exists() and (d / f"{d.name}_CD.json").exists())
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    print(f"extracting {CHIMERA_TILES_PER_SLIDE} tiles/slide from {len(case_dirs)} chimera cases with {workers} workers", flush=True)
    started = time.monotonic()
    all_rows = []
    with mp.Pool(workers) as pool:
        for done, (case_id, rows) in enumerate(pool.imap_unordered(_chimera_extract_one, case_dirs), start=1):
            all_rows.extend(rows)
            if done % 10 == 0 or done == len(case_dirs):
                print(f"[{done}/{len(case_dirs)}] {time.monotonic()-started:.0f}s elapsed, {len(all_rows):,} tiles", flush=True)
    table = pa.table({
        "jpeg": [r[0] for r in all_rows],
        "slide_id": [r[1] for r in all_rows],
        "cohort": [r[2] for r in all_rows],
        "progression": pa.array([r[3] for r in all_rows], type=pa.int8()),
    })
    pq.write_table(table, out_data / "chimera-00000.parquet", compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
    labels = sorted({(r[1], r[2], r[3]) for r in all_rows})
    (tile_dir / "labels.csv").write_text("slide_id,cohort,progression\n" + "\n".join(f"{s},{c},{p}" for s, c, p in labels) + "\n")
    print(f"wrote {len(all_rows):,} tiles across {len(labels)} cases to {out_data}", flush=True)


# fetch_chimera is the standard prepare.py download=True hook. The full pipeline is too slow
# for an interactive prep run (~1 hr download + ~10 min extraction), so we route the user to
# the dedicated sbatch and only check that the cache is populated.
def fetch_chimera(root):
    raise SystemExit(
        f"chimera tile cache is built by a separate sbatch (download is ~350 GB, takes ~1.5 hr). "
        f"Run `sbatch submit/prepare_chimera.sbatch` to populate {root}; then re-run "
        f"`python prepare.py <config> download=False` to verify."
    )


# SurGen (Myles et al. 2025, https://arxiv.org/abs/2502.04946) SR1482 cohort: 416 NMIBC...
# wait, SurGen SR1482 is colorectal cancer, not bladder. 416 cases × ~4 GB .czi each = 2.3 TiB
# of WSIs on the public path-datasets bucket. We can't sync-then-extract like chimera (1.6 TB
# of usable slides won't comfortably stage on /data) — so the prep job streams: download one
# .czi per worker, sample 256 random tissue tiles, JPEG-encode, then DELETE the .czi before
# moving to the next case. Net transient disk peak ≈ N_WORKERS × ~4 GB = ~16 GB. CZI mosaic
# reads use aicspylibczi (slides are M-plane mosaics, not pyramid TIFFs).
SURGEN_BUCKET = "s3://path-datasets/SurGen/SR1482_WSIs"
SURGEN_LABELS_KEY = "s3://path-datasets/SurGen/SR1482_labels.csv"
SURGEN_LEVEL0_TILE = 448  # 224 px × 2 to read 0.50 mpp from the 0.25 mpp 40x level-0 mosaic.
SURGEN_OUT_TILE = 224
SURGEN_TILES_PER_SLIDE = 256
SURGEN_RNG_SEED = 1337
SURGEN_THUMB_SCALE = 0.05  # cheap thumbnail for tissue masking via Otsu.
SURGEN_TISSUE_THRESHOLD = 0.7  # require tile center mean(R,G,B) brightness < this fraction (1.0=white).


# Cohort filter — extended-RAS = (KRAS or NRAS) mutated. KRAS=='No mutation' AND NRAS=='No
# mutation' → wt (label 0); either column with 'p.' or 'c.' notation → mut (label 1); rows
# with Failed/Insufficient/Not performed in both → dropped. Returns DataFrame with case_id,
# slide_basename, ras columns; ~398 usable cases at ~48% positive.
def _surgen_extras_cohort(labels_csv_path, slide_basenames):
    import pandas as pd
    m = pd.read_csv(labels_csv_path)
    def lbl(row):
        k, n = str(row.KRAS).strip(), str(row.NRAS).strip()
        k_wt, n_wt = k == "No mutation", n == "No mutation"
        k_mut, n_mut = ("p." in k or "c." in k), ("p." in n or "c." in n)
        if k_wt and n_wt: return 0
        if k_mut or n_mut: return 1
        return None
    m["ras"] = m.apply(lbl, axis=1)
    m = m[m.ras.notna()].copy()
    # Pick canonical slide per case: T{N:03d}_01.czi if present, else _02.
    by_case = {}
    for bn in slide_basenames:
        # SR1482_40X_HE_T002_01.czi -> case "002", block "01"
        parts = bn.replace(".czi", "").split("_")
        case_n, block = parts[-2].lstrip("T"), parts[-1]
        by_case.setdefault(int(case_n), {})[block] = bn
    rows = []
    for _, row in m.iterrows():
        cid = int(row.case_id)
        blocks = by_case.get(cid, {})
        bn = blocks.get("01") or blocks.get("02")
        if bn is not None:
            rows.append({"case_id": f"T{cid:03d}", "slide_basename": bn, "ras": int(row.ras)})
    return pd.DataFrame(rows)


# Pool worker: download one .czi, mosaic-stitch, sample 256 tissue tiles, JPEG-encode, delete.
# `args` is a (case_id, slide_basename, ras_label, raw_dir) tuple.
def _surgen_extract_one(args):
    import io, numpy as np
    from PIL import Image
    from aicspylibczi import CziFile
    case_id, slide_basename, ras, raw_dir = args
    czi_path = raw_dir / slide_basename
    subprocess.run(["aws", "s3", "cp", f"{SURGEN_BUCKET}/{slide_basename}", str(czi_path), "--quiet"], check=True)
    czi = CziFile(str(czi_path))
    bbox = czi.get_mosaic_bounding_box()
    # Otsu threshold on a low-res thumbnail (mean of RGB) — keep darker pixels (= tissue).
    thumb = czi.read_mosaic(region=(bbox.x, bbox.y, bbox.w, bbox.h), scale_factor=SURGEN_THUMB_SCALE, C=0)[0]
    gray = thumb.mean(axis=-1).astype(np.float32) / 255.0
    # tissue = brightness below threshold (white background → 1.0, tissue → ~0.5-0.8)
    fg = np.argwhere(gray < SURGEN_TISSUE_THRESHOLD)
    rng = np.random.default_rng(SURGEN_RNG_SEED + int(case_id.lstrip("T")))
    rows = []
    if len(fg) > 0:
        n = min(SURGEN_TILES_PER_SLIDE, len(fg))
        chosen = rng.choice(len(fg), size=n, replace=False)
        # mosaic coords are level-0 absolute (with bbox.x/.y as offsets from origin).
        thumb_h, thumb_w = gray.shape
        scale_x = bbox.w / thumb_w
        scale_y = bbox.h / thumb_h
        for i in chosen:
            ty, tx = fg[i]
            cx = int(bbox.x + (tx + 0.5) * scale_x)
            cy = int(bbox.y + (ty + 0.5) * scale_y)
            x0 = max(bbox.x, min(bbox.x + bbox.w - SURGEN_LEVEL0_TILE, cx - SURGEN_LEVEL0_TILE // 2))
            y0 = max(bbox.y, min(bbox.y + bbox.h - SURGEN_LEVEL0_TILE, cy - SURGEN_LEVEL0_TILE // 2))
            tile = czi.read_mosaic(region=(x0, y0, SURGEN_LEVEL0_TILE, SURGEN_LEVEL0_TILE), scale_factor=1.0, C=0)[0]
            buf = io.BytesIO()
            Image.fromarray(tile).resize((SURGEN_OUT_TILE, SURGEN_OUT_TILE), Image.BILINEAR).save(buf, "JPEG", quality=95)
            rows.append((buf.getvalue(), case_id, slide_basename.replace(".czi", ""), ras))
    czi_path.unlink()  # free the 4 GB .czi immediately so the next download has room.
    return case_id, rows


def prepare_surgen_tiles(raw_dir, tile_dir):
    """Stream-and-tile: download → tile → delete each .czi, write a single parquet shard + labels.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_data = tile_dir / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    labels_csv = raw_dir / "SR1482_labels.csv"
    subprocess.run(["aws", "s3", "cp", SURGEN_LABELS_KEY, str(labels_csv), "--quiet"], check=True)
    # Enumerate available slide basenames on the bucket (cheap listing, no downloads).
    listing = subprocess.run(["aws", "s3", "ls", SURGEN_BUCKET + "/"], capture_output=True, text=True, check=True).stdout
    basenames = [line.split()[-1] for line in listing.splitlines() if line.strip().endswith(".czi")]
    cohort = _surgen_extras_cohort(labels_csv, basenames)
    print(f"surgen extras cohort: {len(cohort)} cases ({cohort.ras.mean():.3f} positive rate)", flush=True)
    workers = int(os.environ.get("PREPARE_WORKERS", min(8, os.cpu_count() or 4)))
    args = [(r.case_id, r.slide_basename, int(r.ras), raw_dir) for r in cohort.itertuples()]
    started = time.monotonic()
    all_rows = []
    with mp.Pool(workers) as pool:
        for done, (case_id, rows) in enumerate(pool.imap_unordered(_surgen_extract_one, args), start=1):
            all_rows.extend(rows)
            if done % 20 == 0 or done == len(args):
                print(f"[{done}/{len(args)}] {time.monotonic()-started:.0f}s elapsed, {len(all_rows):,} tiles", flush=True)
    table = pa.table({
        "jpeg": [r[0] for r in all_rows],
        "case_id": [r[1] for r in all_rows],
        "slide_id": [r[2] for r in all_rows],
        "ras": pa.array([r[3] for r in all_rows], type=pa.int8()),
    })
    pq.write_table(table, out_data / "surgen-00000.parquet", compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
    labels = sorted({(r[1], r[2], r[3]) for r in all_rows})
    (tile_dir / "labels.csv").write_text("case_id,slide_id,ras\n" + "\n".join(f"{c},{s},{r}" for c, s, r in labels) + "\n")
    print(f"wrote {len(all_rows):,} tiles across {len(labels)} cases to {out_data}", flush=True)


def fetch_surgen(root):
    raise SystemExit(
        f"surgen tile cache is built by a separate sbatch (download is ~1.6 TB streamed via "
        f"download → tile → delete pipeline, takes ~30 min). "
        f"Run `sbatch submit/prepare_surgen.sbatch` to populate {root}; then re-run "
        f"`python prepare.py <config> download=False` to verify."
    )


# CoNSeP also ships as a single zip behind a form on the HoVer-Net authors' page. Same pattern:
# user drops `consep.zip` into the configured root and we unpack it on download=True.
def fetch_consep(root):
    z = root / "consep.zip"
    if z.exists():
        shutil.unpack_archive(z, root)
        z.unlink()
        return
    raise SystemExit(
        f"consep requires manual access from https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/. "
        f"Drop `consep.zip` into:\n  {root}\nThen re-run `python prepare.py <config> download=True` to unpack."
    )


FETCHERS = {
    "bach": fetch_bach,
    "bracs": fetch_bracs,
    "break_his": fetch_break_his,
    "mhist": fetch_mhist,
    "pcam": fetch_pcam,
    "pannuke": fetch_pannuke,
    "monusac": fetch_monusac,
    "consep": fetch_consep,
    "chimera_tiles": fetch_chimera,
    "surgen_tiles": fetch_surgen,
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
# pathorob needs the camelyon + tolkach_esca parquet shards (TCGA isn't used).
# monusac needs both the train + test extracted directories.
def is_populated(name, p):
    if not p.exists() or not any(p.iterdir()):
        return False
    if name == "mhist" and not (p / "images").exists():
        return False
    if name == "monusac" and not ((p / "MoNuSAC_images_and_annotations").exists() and (p / "MoNuSAC Testing Data and Annotations").exists()):
        return False
    if name == "consep" and not ((p / "Train" / "Images").exists() and (p / "Test" / "Images").exists()):
        return False
    if name == "chimera_tiles" and not list((p / "data").glob("chimera-*.parquet")):
        return False
    if name == "surgen_tiles" and not list((p / "data").glob("surgen-*.parquet")):
        return False
    if name == "pathorob":
        for ds in ("camelyon", "tolkach_esca"):
            if not list((p / ds).glob("data/*.parquet")):
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
