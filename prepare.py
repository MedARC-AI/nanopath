# Single data-prep entry point. Reads the YAML config the user passes in and
# checks every path train.py will read:
#   - data.dataset_dir/shard-NNNNN.parquet   (the 4M-tile dataset, sharded)
#   - probe.dataset_roots[name] for each configured probe dataset
#   - Meta's DINOv2 pretrained weights for cfg["model"]["type"] (torch.hub cache)
# Defaults to HF for the tile dataset (medarc/nanopath), official/HF probe
# sources, and dl.fbaipublicfiles.com for DINOv2 weights.
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
import http.client
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
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


HF_REPO_ID = "medarc/nanopath"
HF_PROBE_PREFIX = "probes"
PROBE_ACCESS_NOTICES = {
    "consep": "you MUST satisfy the official CoNSeP/Warwick access terms at https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/ before using these data; this mirror download is only for portable setup.",
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
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=f"{HF_PROBE_PREFIX}/{filename}"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _http_download_if_needed(args):
    url, dst = args
    if not (dst.exists() and dst.stat().st_size > 1_000_000):
        http_download(url, dst)
    return dst.name


def fetch_pannuke(root):
    for fold in (1, 2, 3):
        if all((root / f"Fold{fold}/{kind}/fold{fold}/{kind}.npy").exists() for kind in ("images", "masks")):
            continue
        zip_path = root / f"fold_{fold}.zip"
        http_download(f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()
        if (root / f"Fold{fold}").exists():
            shutil.rmtree(root / f"Fold{fold}")
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


# PathoBench slide tasks are normally run from Trident patch embeddings. The
# tutorial path extracts a full 20x, 512 px, 0-overlap tissue grid, then pools
# every patch feature (`bag_size=None`). Nanopath mirrors that contract with
# lightweight local tilers and only adapts the train/test split into train/val.
PATHOBENCH_TILING_VERSION = "pathobench_20x_512_v1"
PATHOBENCH_TARGET_MPP = 0.5
PATHOBENCH_PATCH_PX = 512


def _openslide_mpp(slide, default=PATHOBENCH_TARGET_MPP):
    props = slide.properties
    if props.get("openslide.mpp-x") and 0.05 <= float(props["openslide.mpp-x"]) <= 5.0:
        return float(props["openslide.mpp-x"])
    if props.get("tiff.XResolution") and 0.05 <= float(props["tiff.XResolution"]) <= 5.0:
        return float(props["tiff.XResolution"])
    if props.get("openslide.objective-power"):
        return 10.0 / float(props["openslide.objective-power"])
    return default


def _openslide_grid_rows(slide, slide_id, image_col="jpeg", default_mpp=PATHOBENCH_TARGET_MPP):
    import io
    w, h = slide.dimensions
    src = round(PATHOBENCH_PATCH_PX * PATHOBENCH_TARGET_MPP / _openslide_mpp(slide, default_mpp))
    thumb = np.asarray(slide.get_thumbnail((512, 512)).convert("RGB")).mean(axis=2) if slide.level_count > 1 else None
    sx, sy = (w / thumb.shape[1], h / thumb.shape[0]) if thumb is not None else (None, None)
    rows = []
    for y in range(0, max(1, h - src + 1), src):
        for x in range(0, max(1, w - src + 1), src):
            if thumb is not None:
                cy, cx = min(thumb.shape[0] - 1, int((y + src / 2) / sy)), min(thumb.shape[1] - 1, int((x + src / 2) / sx))
                if thumb[cy, cx] >= 230:
                    continue
            elif np.asarray(slide.read_region((max(0, x + src // 2 - 16), max(0, y + src // 2 - 16)), 0, (32, 32)).convert("RGB")).mean() >= 230:
                continue
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
    base = "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0082-pennycuick-lesions/20200517-ftp"
    splits = json.loads((Path(__file__).resolve().parent / "benchmarking" / "ucla_lung.json").read_text())
    slide_ids = sorted(set(splits["train"]["slide_ids"] + splits["val"]["slide_ids"]))
    wsi_dir, slide_cache = root / "wsi", root / UCLA_LUNG_TILING_VERSION
    wsi_dir.mkdir(parents=True, exist_ok=True)
    slide_cache.mkdir(parents=True, exist_ok=True)
    for sid in slide_ids:
        ndpi = wsi_dir / f"{sid}.ndpi"
        if not (ndpi.exists() and ndpi.stat().st_size > 1_000_000):
            http_download(f"{base}/{sid}.ndpi", ndpi)
    workers = int(os.environ.get("PREPARE_WORKERS", min(16, os.cpu_count() or 8)))
    with mp.Pool(workers) as pool:
        args = [(str(wsi_dir / f"{sid}.ndpi"), sid, str(slide_cache)) for sid in slide_ids]
        for done, (sid, n) in enumerate(pool.imap_unordered(_ucla_lung_extract_one, args), start=1):
            if done % 16 == 0 or done == len(args):
                print(f"  [{done}/{len(args)}] {sid}: {n:,} tiles", flush=True)
    table = pa.concat_tables([pq.read_table(slide_cache / f"{sid}.parquet") for sid in slide_ids])
    pq.write_table(table, root / "tiles.parquet", compression="none", row_group_size=PARQUET_ROW_GROUP_SIZE)
    (root / "tiles.parquet").chmod(0o664)
    version = root / "tiling_version.txt"
    version.write_text(UCLA_LUNG_TILING_VERSION + "\n")
    version.chmod(0o664)


def _tile_her2_slide(args):
    slide_id, src, dst = args
    if dst.exists():
        n = sum(1 for _ in dst.glob("*.jpg"))
        if n > 0:
            return slide_id, n
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(src))
    rows = _openslide_grid_rows(slide, slide_id)
    for n, row in enumerate(rows):
        (dst / f"{n:06d}.jpg").write_bytes(row["jpeg"])
    slide.close()
    return slide_id, len(rows)


# HER2-Tumor-ROIs response probe. TCIA PathDB exposes direct SVS URLs, so we
# download only the PathoBench fold_0 train-derived train/val slides, tile them,
# and leave the raw SVS files cached under root/raw for resumable setup.
HER2_PATHDB_COLLECTION_ID = 533
HER2_TILING_VERSION = PATHOBENCH_TILING_VERSION


def _her2_urls():
    urls, page = {}, 0
    while True:
        with urllib.request.urlopen(f"https://pathdb.cancerimagingarchive.net/listofimages/{HER2_PATHDB_COLLECTION_ID}?_format=json&page={page}") as r:
            data = json.load(r)
        if not data:
            return urls
        for item in data:
            urls[item["imageid"][0]["value"]] = item["field_wsiimage"][0]["url"]
        page += 1


def fetch_her2(root):
    splits = json.loads((Path(__file__).resolve().parent / "benchmarking" / "her2.json").read_text())
    slides = list(splits["train"]["slides"]) + list(splits["val"]["slides"])
    urls, raw_dir = _her2_urls(), root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    version = root / "tiling_version.txt"
    if (not version.exists() or version.read_text().strip() != HER2_TILING_VERSION) and (root / "tiles").exists():
        shutil.rmtree(root / "tiles")
    workers = int(os.environ.get("PREPARE_WORKERS", min(16, os.cpu_count() or 8)))
    with mp.Pool(workers) as pool:
        for i, name in enumerate(pool.imap_unordered(_http_download_if_needed, [(urls[sid], raw_dir / f"{sid}.svs") for sid in slides]), start=1):
            if i % 8 == 0 or i == len(slides):
                print(f"  [{i}/{len(slides)}] raw {name}", flush=True)
    args = [(sid, raw_dir / f"{sid}.svs", root / "tiles" / sid) for sid in slides]
    with mp.Pool(workers) as pool:
        for i, (sid, n) in enumerate(pool.imap_unordered(_tile_her2_slide, args), start=1):
            print(f"  [{i}/{len(args)}] {sid}: {n} tiles", flush=True)
    version.write_text(HER2_TILING_VERSION + "\n")
    version.chmod(0o664)


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


# PFS_VALENTINO survival probe. Only train/val slides from the vendored split
# are tiled; PathoBench fold_0 test remains held out.
CRC_SURVIVAL_HF_TSV = "crc_outcomes/PFS_VALENTINO/k=all.tsv"
CRC_SURVIVAL_BIOSTUDIES = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/407/S-BIAD1407/Files/VALENTINO"
CRC_SURVIVAL_TILING_VERSION = PATHOBENCH_TILING_VERSION


def _tile_one_crc_survival(args):
    import openslide
    wsi_path, slide_id, cache_dir = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    if cache_path.exists():
        return slide_id, pq.read_metadata(cache_path).num_rows
    slide = openslide.OpenSlide(str(wsi_path))
    # BioStudies VALENTINO TIFF metadata encodes native resolution inconsistently;
    # the cohort is ~0.22 um/px, so use that fixed fallback when needed.
    rows = _openslide_grid_rows(slide, slide_id, image_col="image", default_mpp=0.22)
    slide.close()
    tmp_cache = cache_path.with_suffix(".parquet.part")
    pq.write_table(pa.table({"slide_id": [r["slide_id"] for r in rows], "image": [r["image"] for r in rows]}), tmp_cache, compression="snappy", row_group_size=PARQUET_ROW_GROUP_SIZE)
    os.replace(tmp_cache, cache_path)
    return slide_id, len(rows)


def fetch_crc_survival(root):
    from huggingface_hub import hf_hub_download
    tsv_src = hf_hub_download(repo_id="MahmoodLab/Patho-Bench", repo_type="dataset", filename=CRC_SURVIVAL_HF_TSV)
    shutil.copy(tsv_src, root / "labels.tsv")
    splits = json.loads((Path(__file__).resolve().parent / "benchmarking" / "crc_survival.json").read_text())
    slide_ids = sorted(set(splits["train"]["slide_ids"] + splits["val"]["slide_ids"]))
    wsi_dir, slide_cache = root / "wsi", root / "slides"
    wsi_dir.mkdir(parents=True, exist_ok=True)
    slide_cache.mkdir(parents=True, exist_ok=True)
    version, building = root / "tiling_version.txt", root / "tiling_in_progress.txt"
    if not version.exists() or version.read_text().strip() != CRC_SURVIVAL_TILING_VERSION:
        if not building.exists() or building.read_text().strip() != CRC_SURVIVAL_TILING_VERSION:
            if (root / "patches.parquet").exists():
                (root / "patches.parquet").unlink()
            shutil.rmtree(slide_cache)
            slide_cache.mkdir(parents=True, exist_ok=True)
            building.write_text(CRC_SURVIVAL_TILING_VERSION + "\n")
    for sid in slide_ids:
        out = wsi_dir / f"{sid}.tif"
        if not (out.exists() and out.stat().st_size > 1_000_000):
            http_download(f"{CRC_SURVIVAL_BIOSTUDIES}/{sid}.tif", out.with_suffix(".tif.part"))
            out.with_suffix(".tif.part").rename(out)
    workers = int(os.environ.get("PREPARE_WORKERS", os.cpu_count() or 8))
    tiles = 0
    with mp.Pool(workers) as pool:
        for done, (sid, n) in enumerate(pool.imap_unordered(_tile_one_crc_survival, [(wsi_dir / f"{sid}.tif", sid, str(slide_cache)) for sid in slide_ids]), start=1):
            tiles += n
            if done % 10 == 0 or done == len(slide_ids):
                print(f"  [{done}/{len(slide_ids)}] patches={tiles:,}", flush=True)
    table = pa.concat_tables([pq.read_table(slide_cache / f"{sid}.parquet") for sid in slide_ids])
    pq.write_table(table, root / "patches.parquet", compression="snappy", row_group_size=PARQUET_ROW_GROUP_SIZE)
    version.write_text(CRC_SURVIVAL_TILING_VERSION + "\n")
    version.chmod(0o664)
    building.unlink(missing_ok=True)


# PathoROB ships as two HF datasets; TCGA subset is intentionally excluded.
def fetch_pathorob(root):
    from huggingface_hub import snapshot_download
    for subset in ("camelyon", "tolkach_esca"):
        snapshot_download(repo_id=f"bifold-pathomics/PathoROB-{subset}", repo_type="dataset", local_dir=str(root / subset))


def fetch_monusac(root):
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
    "crc_survival": fetch_crc_survival,
    "her2": fetch_her2,
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
        rel = json.loads((bench / f"{name}.json").read_text())["train"]["images"][0]
        return (p / rel).exists()
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
    if name == "her2":
        splits = json.loads((bench / "her2.json").read_text())
        slides = list(splits["train"]["slides"]) + list(splits["val"]["slides"])
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == HER2_TILING_VERSION and all(any((p / "tiles" / sid).glob("*.jpg")) for sid in slides)
    if name == "surgen":
        splits = json.loads((bench / "surgen.json").read_text())
        expected = set(splits["train"]["slides"] + splits["val"]["slides"])
        files = sorted((p / "data").glob("surgen-*.parquet"))
        labels = {line.split(",")[0] for line in (p / "labels.csv").read_text().splitlines()[1:]} if (p / "labels.csv").exists() else set()
        got = set(pa.concat_tables([pq.read_table(f, columns=["slide_id"]) for f in files]).column("slide_id").to_pylist()) if files else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == SURGEN_TILING_VERSION and expected <= labels and expected <= got
    if name == "crc_survival":
        splits = json.loads((bench / "crc_survival.json").read_text())
        expected = set(splits["train"]["slide_ids"] + splits["val"]["slide_ids"])
        got = set(pq.read_table(p / "patches.parquet", columns=["slide_id"]).column("slide_id").to_pylist()) if (p / "patches.parquet").exists() else set()
        version = p / "tiling_version.txt"
        return version.exists() and version.read_text().strip() == CRC_SURVIVAL_TILING_VERSION and (p / "labels.tsv").exists() and expected <= got
    if name == "pathorob" and not all(list((p / s / "data").glob("*.parquet")) for s in ("camelyon", "tolkach_esca")):
        return False
    if name == "monusac" and not any((p / "MoNuSAC_images_and_annotations").glob("*/*.npy")):
        return False
    if name == "consep" and not ((p / "Train" / "Images").exists() and (p / "Train" / "Labels").exists()):
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
    if download:
        for name in cfg["probe"]["dataset_roots"]:
            if name in PROBE_ACCESS_NOTICES:
                print(f"[notice] probe/{name}: {PROBE_ACCESS_NOTICES[name]}", flush=True)
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
            lines.append(f"  probe/{name}: {root} is empty, missing, or stale for the current benchmark")
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

    # Reaching here means tiles + every configured probe dataset + DINOv2 weights are
    # in place. Tell the user explicitly so they don't have to read between
    # the [skip] lines.
    n_shards = sum(1 for _ in dataset_dir.glob("shard-*.parquet"))
    n_probes = len(cfg["probe"]["dataset_roots"])
    print(
        f"\nAll data ready: {n_shards} parquet shards at {dataset_dir}, {n_probes} probe datasets "
        f"({', '.join(cfg['probe']['dataset_roots'])}), and {cfg['model']['type']} weights at "
        f"{weights_path}. Launch training with `python train.py {config_path}` or "
        f"`sbatch submit/train_1gpu.sbatch {config_path}`.",
        flush=True,
    )


if __name__ == "__main__":
    main()
