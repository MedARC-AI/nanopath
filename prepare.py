# Single data-prep entry point. Reads configs/main.yaml by default (or a
# user-passed YAML config) and checks every path train.py will read:
#   - data.dataset_dir/shard-NNNNN.parquet   (the 4M-tile dataset, sharded)
#   - probe.dataset_roots[name] for each configured probe dataset
#   - Meta's DINOv2 pretrained weights for cfg["model"]["type"] (torch.hub cache)
# Defaults to HF for the tile dataset and probe assets. Official-source helper
# functions are maintainer rebuild paths for creating the HF probe mirrors.
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

import http.client
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openslide
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent
HF_REPO_ID = "medarc/nanopath"
HF_PROBE_PREFIX = "probes"
PROBE_ACCESS_NOTICES = {
    "consep": "you MUST satisfy the official CoNSeP/Warwick access terms at https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/ before using these data; this mirror download is only for portable setup.",
    "leopard_bcr": "LEOPARD BCR is rehosted in medarc/nanopath with organizer permission; see https://leopard.grand-challenge.org/ for the upstream challenge.",
    "mhist": "you MUST complete MHIST's Dataset Research Use Agreement at https://bmirds.github.io/MHIST/ before using these data; this mirror download is only for portable setup.",
}
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


# Fetch the expected byte count so resumable WSI prep can reject truncated files.
def http_size(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nanopath"}, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


# Stream a URL to disk in chunks so large probe archives do not sit in memory.
def http_download(url, dst):
    if dst.exists() and dst.stat().st_size > 0: print(f"  [skip] {dst}", flush=True); return
    print(f"  GET {url}\n   -> {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    expected = None
    while expected is None or tmp.stat().st_size < expected:
        before = tmp.stat().st_size if tmp.exists() else 0
        for attempt in range(20):
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "nanopath", **({"Range": f"bytes={offset}-"} if offset else {})}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resumed = bool(r.headers.get("Content-Range"))
                    if offset and not resumed:
                        offset = before = 0
                    with tmp.open("ab" if offset else "wb") as f:
                        if resumed:
                            expected = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                        elif expected is None and r.headers.get("Content-Length"):
                            expected = offset + int(r.headers["Content-Length"])
                        shutil.copyfileobj(r, f, length=1 << 20)
                break
            except (http.client.RemoteDisconnected, http.client.IncompleteRead, urllib.error.URLError, ConnectionResetError, TimeoutError):
                print(f"  retry {attempt + 1}/20: {tmp}", flush=True)
                time.sleep(min(60, 2 + attempt))
        if tmp.stat().st_size == before:
            print(f"  retry stalled: {tmp}", flush=True)
            time.sleep(60)
            continue
        if expected is None:
            break
    if expected is not None:
        assert tmp.stat().st_size == expected, f"truncated download: {tmp} has {tmp.stat().st_size} bytes, expected {expected}"
    os.replace(tmp, dst)


def hf_download(filename, dst):
    if dst.exists() and dst.stat().st_size > 0: print(f"  [skip] {dst}", flush=True); return
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=f"{HF_PROBE_PREFIX}/{filename}"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def hf_probe_dir(name, root):
    from huggingface_hub import snapshot_download
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    print(f"  using medarc/nanopath probe mirror: {name}", flush=True)
    snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(root), allow_patterns=[f"{HF_PROBE_PREFIX}/{name}/**"], max_workers=workers)
    src = root / HF_PROBE_PREFIX / name
    for p in src.iterdir():
        dst = root / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(p), dst)
    shutil.rmtree(root / HF_PROBE_PREFIX)


def make_hpcroot_writable(root):
    import grp
    gid = grp.getgrnam("hpcroot").gr_gid
    for p in [root, *root.rglob("*")]:
        os.chown(p, -1, gid)
        p.chmod(p.stat().st_mode | 0o660 | (0o110 if p.is_dir() else 0))


def fetch_pannuke(root):
    for fold in (1, 2):
        if all((root / f"Fold{fold}/{kind}/fold{fold}/{kind}.npy").exists() for kind in ("images", "masks")):
            continue
        zip_path = root / f"fold_{fold}.zip"
        hf_download(f"pannuke/fold_{fold}.zip", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()
        spaced = root / f"Fold {fold}"
        if spaced.exists():
            if (root / f"Fold{fold}").exists():
                shutil.rmtree(root / f"Fold{fold}")
            spaced.rename(root / f"Fold{fold}")


def fetch_pcam(root):
    hf_probe_dir("pcam", root)


def fetch_bracs(root):
    hf_probe_dir("bracs", root)


def fetch_break_his(root):
    hf_probe_dir("break_his", root)


# PathoBench slide tasks are normally run from Trident patch embeddings. The
# tutorial path extracts a full 20x, 512 px, 0-overlap tissue grid, then pools
# every patch feature (`bag_size=None`). Nanopath mirrors that contract with
# lightweight local tilers and only adapts the train/test split into train/val.
PATHOBENCH_TILING_VERSION = "pathobench_20x_512_v1"
PATHOBENCH_TARGET_MPP = 0.5
PATHOBENCH_PATCH_PX = 512
CPTAC_PDA_OS_TILING_VERSION = PATHOBENCH_TILING_VERSION + "_cptac_pda_os_fold0_train_v1"
CPTAC_PDA_OS_RAW_BASE = "https://pathdb.cancerimagingarchive.net/system/files/wsi/ross/CPTAC/PDA"


def _openslide_mpp(slide, default=PATHOBENCH_TARGET_MPP):
    props = slide.properties
    if props.get("openslide.mpp-x") and 0.05 <= float(props["openslide.mpp-x"]) <= 5.0:
        return float(props["openslide.mpp-x"])
    if props.get("tiff.XResolution") and 0.05 <= float(props["tiff.XResolution"]) <= 5.0:
        return float(props["tiff.XResolution"])
    if props.get("openslide.objective-power"):
        return 10.0 / float(props["openslide.objective-power"])
    return default


def _openslide_grid_rows(slide, slide_id, image_col="jpeg", default_mpp=PATHOBENCH_TARGET_MPP, cap=0):
    import io
    w, h = slide.dimensions
    src = round(PATHOBENCH_PATCH_PX * PATHOBENCH_TARGET_MPP / _openslide_mpp(slide, default_mpp))
    thumb = np.asarray(slide.get_thumbnail((512, 512)).convert("RGB")).mean(axis=2) if slide.level_count > 1 else None
    sx, sy = (w / thumb.shape[1], h / thumb.shape[0]) if thumb is not None else (None, None)
    coords = []
    for y in range(0, max(1, h - src + 1), src):
        for x in range(0, max(1, w - src + 1), src):
            if thumb is not None:
                cy, cx = min(thumb.shape[0] - 1, int((y + src / 2) / sy)), min(thumb.shape[1] - 1, int((x + src / 2) / sx))
                if thumb[cy, cx] >= 230:
                    continue
            elif np.asarray(slide.read_region((max(0, x + src // 2 - 16), max(0, y + src // 2 - 16)), 0, (32, 32)).convert("RGB")).mean() >= 230:
                continue
            coords.append((x, y))
    if cap and len(coords) > cap:
        coords = [coords[int(i)] for i in np.linspace(0, len(coords) - 1, cap, dtype=np.int64)]
    rows = []
    for x, y in coords:
        tile = slide.read_region((x, y), 0, (src, src)).convert("RGB").resize((PATHOBENCH_PATCH_PX, PATHOBENCH_PATCH_PX), Image.BILINEAR)
        buf = io.BytesIO()
        tile.save(buf, "JPEG", quality=JPEG_QUALITY)
        rows.append({"slide_id": slide_id, image_col: buf.getvalue()})
    return rows


# UCLA Lung (idr0082) slide-level progression/regression probe.
UCLA_LUNG_TILING_VERSION = PATHOBENCH_TILING_VERSION


def _ucla_lung_extract_one(args):
    import openslide
    ndpi_path, slide_id, cache_dir = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    if cache_path.exists():
        return slide_id, pq.read_metadata(cache_path).num_rows
    slide = openslide.OpenSlide(ndpi_path)
    rows = _openslide_grid_rows(slide, slide_id)
    for i, row in enumerate(rows):
        row["tile_idx"] = i
    slide.close()
    tmp_cache = cache_path.with_suffix(".parquet.part")
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in ("slide_id", "tile_idx", "jpeg")}), tmp_cache, compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
    os.replace(tmp_cache, cache_path)
    cache_path.chmod(0o664)
    return slide_id, len(rows)


def fetch_ucla_lung(root):
    hf_probe_dir("ucla_lung", root)


# PathoBench SR386 RAS mutation probe. Normal setup pulls our pre-extracted
# HF parquet mirror; this official-source path rebuilds that mirror by streaming
# CZI files, caching one parquet per slide, then deleting raw CZI.
SURGEN_EBI_BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/SR386_WSIs"
SURGEN_TILING_VERSION = PATHOBENCH_TILING_VERSION
SURGEN_THUMB_SCALE = 0.01
SURGEN_TISSUE_BAND = (0.1, 0.85)
SURGEN_HF_SHARDS = 16


def _surgen_extract_one(args):
    import io
    from aicspylibczi import CziFile
    slide_id, ras, raw_dir, cache_dir = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    if cache_path.exists():
        return slide_id, pq.read_metadata(cache_path).num_rows
    czi_path = Path(raw_dir) / f"{slide_id}.czi"
    url = f"{SURGEN_EBI_BASE}/{slide_id}.czi"
    expected = http_size(url)
    if not czi_path.exists() or czi_path.stat().st_size != expected:
        http_download(url, czi_path)
    assert czi_path.stat().st_size == expected and expected > 100_000_000, f"bad SurGen CZI: {czi_path}"
    czi = CziFile(str(czi_path))
    bbox = czi.get_mosaic_bounding_box()
    md = ET.tostring(czi.meta, encoding="unicode")
    mpp = float(re.search(r'<Distance Id="X">\s*<Value>([^<]+)</Value>', md).group(1)) * 1e6
    scale = mpp / PATHOBENCH_TARGET_MPP
    src_tile = round(PATHOBENCH_PATCH_PX / scale)
    thumb = czi.read_mosaic(region=(bbox.x, bbox.y, bbox.w, bbox.h), scale_factor=SURGEN_THUMB_SCALE, C=0)[0]
    gray = thumb.mean(axis=-1).astype(np.float32) / 255.0
    h, w = gray.shape
    lo, hi = SURGEN_TISSUE_BAND
    rows = []
    for cy in range(bbox.y + src_tile // 2, bbox.y + bbox.h - src_tile // 2, src_tile):
        for cx in range(bbox.x + src_tile // 2, bbox.x + bbox.w - src_tile // 2, src_tile):
            ty, tx = min(h - 1, int((cy - bbox.y) / bbox.h * h)), min(w - 1, int((cx - bbox.x) / bbox.w * w))
            if lo < gray[ty, tx] < hi:
                tile = czi.read_mosaic(region=(cx - src_tile // 2, cy - src_tile // 2, src_tile, src_tile), scale_factor=scale, C=0)[0]
                buf = io.BytesIO()
                Image.fromarray(tile).resize((PATHOBENCH_PATCH_PX, PATHOBENCH_PATCH_PX), Image.BILINEAR).save(buf, "JPEG", quality=JPEG_QUALITY)
                rows.append((buf.getvalue(), slide_id, ras))
    tmp_cache = cache_path.with_suffix(".parquet.part")
    pq.write_table(
        pa.table({"jpeg": [r[0] for r in rows], "slide_id": [r[1] for r in rows], "ras": pa.array([r[2] for r in rows], type=pa.int8())}),
        tmp_cache,
        compression="none",
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )
    os.replace(tmp_cache, cache_path)
    czi_path.unlink()
    return slide_id, len(rows)


def fetch_surgen(root):
    # Default user path: pull the already extracted 20x/512 train-fold tile cache.
    # Rebuild this HF mirror with fetch_surgen_from_official_sources() when the
    # split or tiling recipe changes.
    from huggingface_hub import snapshot_download
    print("  using pre-extracted SurGen tile cache from medarc/nanopath; official EBI CZI regeneration is multi-hour", flush=True)
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    (root / "data").mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(root), allow_patterns=["probes/surgen/*"], max_workers=workers)
    src = root / HF_PROBE_PREFIX / "surgen"
    for f in (root / "data").glob("surgen-*.parquet"):
        f.unlink()
    for f in sorted(src.glob("surgen-*.parquet")):
        os.replace(f, root / "data" / f.name)
    os.replace(src / "labels.csv", root / "labels.csv")
    os.replace(src / "tiling_version.txt", root / "tiling_version.txt")
    shutil.rmtree(root / HF_PROBE_PREFIX)


def fetch_surgen_from_official_sources(root):
    raw, out_data, slide_cache = root / "raw", root / "data", root / "slides"
    raw.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    slide_cache.mkdir(parents=True, exist_ok=True)
    version, building = root / "tiling_version.txt", root / "tiling_in_progress.txt"
    if not version.exists() or version.read_text().strip() != SURGEN_TILING_VERSION:
        if not building.exists() or building.read_text().strip() != SURGEN_TILING_VERSION:
            shutil.rmtree(out_data)
            shutil.rmtree(slide_cache)
            out_data.mkdir(parents=True, exist_ok=True)
            slide_cache.mkdir(parents=True, exist_ok=True)
            building.write_text(SURGEN_TILING_VERSION + "\n")
    splits = json.loads((Path(__file__).resolve().parent / "benchmarking" / "surgen.json").read_text())
    cohort = [(sid, int(lbl), str(raw), str(slide_cache)) for split in ("train", "val") for sid, lbl in zip(splits[split]["slides"], splits[split]["labels"])]
    workers = int(os.environ.get("PREPARE_WORKERS", min(8, os.cpu_count() or 4)))
    tiles = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, fut in enumerate(as_completed(pool.submit(_surgen_extract_one, x) for x in cohort), start=1):
            sid, n = fut.result()
            tiles += n
            if done % 20 == 0 or done == len(cohort):
                print(f"  [{done}/{len(cohort)}] {tiles:,} tiles", flush=True)
    for f in out_data.glob("surgen-*.parquet"):
        f.unlink()
    chunk = (len(cohort) + SURGEN_HF_SHARDS - 1) // SURGEN_HF_SHARDS
    for i in range(0, len(cohort), chunk):
        table = pa.concat_tables([pq.read_table(slide_cache / f"{sid}.parquet") for sid, *_ in cohort[i : i + chunk]])
        out = out_data / f"surgen-{i // chunk:05d}.parquet"
        tmp = out.with_suffix(".parquet.part")
        pq.write_table(table, tmp, compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
        os.replace(tmp, out)
    labels = sorted((sid, ras) for sid, ras, *_ in cohort)
    (root / "labels.csv").write_text("slide_id,ras\n" + "\n".join(f"{s},{r}" for s, r in labels) + "\n")
    version.write_text(SURGEN_TILING_VERSION + "\n")
    version.chmod(0o664)
    building.unlink(missing_ok=True)


# LEOPARD biochemical-recurrence survival probe. The checked-in JSON keeps all
# recurrence events plus the longest-follow-up censored controls; source prep
# streams official S3 slides one at a time into a compact capped tile cache.
LEOPARD_BCR_TILING_VERSION = PATHOBENCH_TILING_VERSION + "_leopard_bcr_174x768_v1"
LEOPARD_BCR_TILES_PER_SLIDE = 768


def _leopard_bcr_extract_one(args):
    import openslide
    case_id, slide_id, event, days, raw_dir, cache_dir = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    tif_path = Path(raw_dir) / f"{slide_id}.tif"
    if cache_path.exists():
        tif_path.unlink(missing_ok=True)
        return slide_id, pq.read_metadata(cache_path).num_rows
    url = f"https://leopard-challenge.s3.us-west-2.amazonaws.com/training/{slide_id}.tif"
    expected = http_size(url)
    if tif_path.exists() and tif_path.stat().st_size != expected:
        tif_path.unlink()
    http_download(url, tif_path)
    assert tif_path.stat().st_size == expected, f"bad LEOPARD TIFF: {tif_path}"
    slide = openslide.OpenSlide(str(tif_path))
    rows = _openslide_grid_rows(slide, slide_id, image_col="image", cap=LEOPARD_BCR_TILES_PER_SLIDE)
    slide.close()
    for i, row in enumerate(rows):
        row["case_id"], row["tile_idx"] = case_id, i
    tmp = cache_path.with_suffix(".parquet.part")
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in ("case_id", "slide_id", "tile_idx", "image")}), tmp, compression="snappy", row_group_size=PARQUET_ROW_GROUP_SIZE)
    os.replace(tmp, cache_path)
    tif_path.unlink()
    return slide_id, len(rows)


def fetch_leopard_bcr(root):
    hf_probe_dir("leopard_bcr", root)
    if root == Path("/data/leopard_bcr"):
        make_hpcroot_writable(root)


def fetch_leopard_bcr_from_official_sources(root):
    splits = json.loads((REPO_ROOT / "benchmarking" / "leopard_bcr.json").read_text())
    raw_dir, slide_cache = root / "raw", root / "slides"
    raw_dir.mkdir(parents=True, exist_ok=True)
    slide_cache.mkdir(parents=True, exist_ok=True)
    version = root / "tiling_version.txt"
    if version.exists() and version.read_text().strip() != LEOPARD_BCR_TILING_VERSION:
        for stale in ("patches.parquet", "labels.tsv"):
            (root / stale).unlink(missing_ok=True)
        shutil.rmtree(slide_cache)
        slide_cache.mkdir(parents=True)
    cohort = [(case, slides[0], event, days, str(raw_dir), str(slide_cache)) for case, slides, event, days in zip(splits["case_ids"], splits["case_slides"], splits["events"], splits["days"])]
    workers = min(4, int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 4)))
    print(f"  rebuilding LEOPARD BCR tile cache from official S3 TIFFs ({len(cohort)} slides, {workers} workers)", flush=True)
    with mp.Pool(workers) as pool:
        for done, (sid, n) in enumerate(pool.imap_unordered(_leopard_bcr_extract_one, cohort), start=1):
            if done % 10 == 0 or done == len(cohort):
                print(f"  [{done}/{len(cohort)}] latest={sid} tiles={n:,}", flush=True)
    out_path = root / "patches.parquet"
    tmp_path = out_path.with_suffix(".parquet.part")
    writer = None
    for _, sid, *_ in cohort:
        table = pq.read_table(slide_cache / f"{sid}.parquet")
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
        writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
    writer.close()
    os.replace(tmp_path, out_path)
    (root / "labels.tsv").write_text("case_id\tslide_id\tBCR_event\tBCR_days\n" + "\n".join(f"{case}\t{sid}\t{event}\t{day}" for case, sid, event, day, *_ in cohort) + "\n")
    version.write_text(LEOPARD_BCR_TILING_VERSION + "\n")
    version.chmod(0o664)
    if root == Path("/data/leopard_bcr"):
        make_hpcroot_writable(root)


def _cptac_pda_os_extract_one(args):
    import openslide
    case_id, slide_id, event, days, raw_dir, cache_dir = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    if cache_path.exists():
        return slide_id, pq.read_metadata(cache_path).num_rows
    svs_path = Path(raw_dir) / f"{slide_id}.svs"
    http_download(f"{CPTAC_PDA_OS_RAW_BASE}/{slide_id}.svs", svs_path)
    slide = openslide.OpenSlide(str(svs_path))
    rows = _openslide_grid_rows(slide, slide_id, image_col="image")
    slide.close()
    for i, row in enumerate(rows):
        row["case_id"], row["tile_idx"] = case_id, i
    tmp = cache_path.with_suffix(".parquet.part")
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in ("case_id", "slide_id", "tile_idx", "image")}), tmp, compression="snappy", row_group_size=PARQUET_ROW_GROUP_SIZE)
    os.replace(tmp, cache_path)
    svs_path.unlink()
    return slide_id, len(rows)


def fetch_cptac_pda_os(root):
    hf_probe_dir("cptac_pda_os", root)


def fetch_cptac_pda_os_from_official_sources(root):
    bench = Path(__file__).resolve().parent / "benchmarking"
    splits = json.loads((bench / "cptac_pda_os.json").read_text())
    raw_dir, slide_cache = root / "raw", root / "slides_full"
    raw_dir.mkdir(parents=True, exist_ok=True)
    slide_cache.mkdir(parents=True, exist_ok=True)
    version = root / "tiling_version.txt"
    if version.exists() and version.read_text().strip() != CPTAC_PDA_OS_TILING_VERSION:
        for stale in ("patches.parquet", "labels.tsv"):
            (root / stale).unlink(missing_ok=True)
        shutil.rmtree(slide_cache)
        slide_cache.mkdir(parents=True)
    events = dict(zip(splits["case_ids"], splits["events"]))
    days = dict(zip(splits["case_ids"], splits["days"]))
    cohort = [(case, sid, events[case], days[case], str(raw_dir), str(slide_cache)) for case, slides in zip(splits["case_ids"], splits["case_slides"]) for sid in slides]
    workers = int(os.environ.get("PREPARE_WORKERS", min(8, os.cpu_count() or 4)))
    print(f"  rebuilding uncapped CPTAC-PDA OS tile cache from TCIA PathDB SVS files ({len(cohort)} slides, {workers} workers)", flush=True)
    with mp.Pool(workers) as pool:
        for done, (sid, n) in enumerate(pool.imap_unordered(_cptac_pda_os_extract_one, cohort), start=1):
            if done % 20 == 0 or done == len(cohort):
                print(f"  [{done}/{len(cohort)}] latest={sid} tiles={n:,}", flush=True)
    out_path = root / "patches.parquet"
    tmp_path = out_path.with_suffix(".parquet.part")
    writer = None
    for _, sid, *_ in cohort:
        table = pq.read_table(slide_cache / f"{sid}.parquet")
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
        writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
    writer.close()
    os.replace(tmp_path, out_path)
    (root / "labels.tsv").write_text("case_id\tslide_id\tOS_event\tOS_days\n" + "\n".join(f"{case}\t{sid}\t{event}\t{day}" for case, sid, event, day, *_ in cohort) + "\n")
    version.write_text(CPTAC_PDA_OS_TILING_VERSION + "\n")


# PathoROB ships as two HF datasets; TCGA subset is intentionally excluded.
def fetch_pathorob(root):
    hf_probe_dir("pathorob", root)


def fetch_monusac(root):
    hf_probe_dir("monusac", root)


def fetch_monusac_from_official_source(root):
    import gdown
    Image.MAX_IMAGE_PIXELS = None
    zip_path = root / "monusac_train.zip"
    gdown.download(id="1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq", output=str(zip_path), quiet=False)
    shutil.unpack_archive(zip_path, root)
    class_id = {"Epithelial": 1, "Lymphocyte": 2, "Macrophage": 3, "Neutrophil": 4}
    for tif in sorted((root / "MoNuSAC_images_and_annotations").glob("*/*.tif")):
        xml_path, npy_path = tif.with_suffix(".xml"), tif.with_suffix(".npy")
        w, h = Image.open(tif).size
        label = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(label)
        for ann in ET.parse(xml_path).getroot().findall(".//Annotation"):
            fill = class_id.get(ann.find("./Attributes/Attribute").get("Name"), 0)
            if fill == 0:
                continue
            for region in ann.findall("./Regions/Region"):
                pts = [(float(v.get("X")), float(v.get("Y"))) for v in region.findall("./Vertices/Vertex")]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=fill)
        np.save(npy_path, np.asarray(label, dtype=np.uint8))


# CoNSeP's Warwick landing page is sign-in gated from batch jobs, so use our
# byte-for-byte probe mirror after warning users about the upstream terms.
def fetch_consep(root):
    zip_path = root / "consep.zip"
    hf_download("consep/consep.zip", zip_path)
    shutil.unpack_archive(zip_path, root)
    for name in ("Train", "Test"):
        if (root / name).exists():
            shutil.rmtree(root / name)
    for p in (root / "CoNSeP").iterdir():
        shutil.move(str(p), root / p.name)
    shutil.rmtree(root / "CoNSeP")
    shutil.rmtree(root / "__MACOSX")


# MHIST's official site is agreement-gated; mirror the exact probe files on HF
# so a fresh `prepare.py ... download=True` run is noninteractive.
def fetch_mhist(root):
    hf_download("mhist/annotations.csv", root / "annotations.csv")
    hf_download("mhist/images.zip", root / "images.zip")
    shutil.unpack_archive(root / "images.zip", root)


FETCHERS = {
    "bracs": fetch_bracs,
    "break_his": fetch_break_his,
    "consep": fetch_consep,
    "cptac_pda_os": fetch_cptac_pda_os,
    "leopard_bcr": fetch_leopard_bcr,
    "mhist": fetch_mhist,
    "monusac": fetch_monusac,
    "pcam": fetch_pcam,
    "pannuke": fetch_pannuke,
    "pathorob": fetch_pathorob,
    "surgen": fetch_surgen,
    "ucla_lung": fetch_ucla_lung,
}


# Resolve $VAR and ~ in a YAML-supplied path string; anything else stays literal.
def _resolve(s):
    return Path(os.path.expanduser(os.path.expandvars(str(s))))


# Missing checked-in /data and /block dataset defaults are shared-cluster paths,
# not portable user choices. Retarget empty/missing ones into repo-local data/
# so `prepare.py ... download=True` makes a fresh clone runnable by itself.
def _local_data_root(s):
    p = _resolve(s)
    if p == Path("/data/leopard_bcr") and Path("/data").exists() and os.access("/data", os.W_OK):
        return str(s)
    if p.is_absolute() and len(p.parts) > 3 and p.parts[1] == "data" and p.parts[3] == "nanopath" and Path("/data").exists() and os.access("/data", os.W_OK):
        return str(s)
    if p.is_absolute() and len(p.parts) > 1 and p.parts[1] in {"data", "block"} and (not p.is_dir() or not any(p.iterdir())):
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
    if not p.exists() or not any(p.iterdir()):
        return False
    bench = Path(__file__).resolve().parent / "benchmarking"
    if name in {"bracs", "break_his", "mhist"}:
        splits = json.loads((bench / f"{name}.json").read_text())
        return all((p / rel).exists() for split in ("train", "val") for rel in splits[split]["images"])
    if name == "pcam":
        return all((p / f"camelyonpatch_level_2_split_{s}_{k}.h5").exists() for s in ("train", "valid") for k in ("x", "y"))
    if name == "pannuke":
        return all((p / f"Fold{fold}/{kind}/fold{fold}/{kind}.npy").exists() for fold in (1, 2) for kind in ("images", "masks"))
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
    if name == "pathorob" and not all(list((p / s / "data").glob("*.parquet")) for s in ("camelyon", "tolkach_esca")):
        return False
    if name == "monusac" and not any((p / "MoNuSAC_images_and_annotations").glob("*/*.npy")):
        return False
    if name == "consep" and not ((p / "Train" / "Images").exists() and (p / "Train" / "Labels").exists()):
        return False
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
        if name in PROBE_ACCESS_NOTICES: print(f"[notice] probe/{name}: {PROBE_ACCESS_NOTICES[name]}", flush=True)
        print(f"[fetch] probe/{name} -> {root}", flush=True)
        FETCHERS[name](root)
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

    # Stage 3 — Meta's pretrained weights for the model variant in cfg
    # (dinov2_vits14_reg ~84 MB, dinov2_vitb14_reg ~330 MB, giant ~4 GB) live in
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
            f"Rerun: {prepare_cmd}"
        )
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] dinov2 weights -> {weights_path}", flush=True)
        torch.hub.load_state_dict_from_url(pretrain_url, model_dir=str(weights_dir), progress=True)
        print("[done] dinov2 weights", flush=True)

    # Reaching here means tiles + every configured probe dataset + DINOv2 weights are
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
