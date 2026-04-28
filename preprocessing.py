# Precompute 224x224 JPEG tiles from the TCGA SVS sample list so train.py can
# stream from JPEGs instead of decoding WSIs every step. Reads the SVS path /
# x / y / level rows from data.sample_list, deterministically subsamples down
# to TARGET_TILE_COUNT (data.split_seed), then writes
#   {data.dataset_dir}/{slide_stem}/{x}_{y}_{level}.jpg
# plus a sorted manifest.txt listing every successfully decoded tile.
# Existing JPEGs are skipped so reruns resume cleanly. Poison slides (open or
# read failures) are logged once per failing slide and skipped; their tiles do
# not appear in the manifest. Run once: `python preprocessing.py
# configs/leader.yaml`.

import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import openslide
import yaml
from PIL import Image


TILE_SIZE = 224
JPEG_QUALITY = 95
TARGET_TILE_COUNT = 4_000_000
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
# whole job: log the first failure per slide to stderr and continue.
def process_row(args):
    dataset_dir, slide_path, x, y, level = args
    rel = f"{Path(slide_path).stem}/{x}_{y}_{level}.jpg"
    out = Path(dataset_dir) / rel
    if out.exists():
        return rel
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
    Image.fromarray(tile).save(out, "JPEG", quality=JPEG_QUALITY)
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


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python preprocessing.py <config.yaml>")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    data = cfg["data"]
    sample_list = Path(data["sample_list"])
    dataset_dir = Path(data["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    total = count_rows(sample_list)
    print(f"sample_list rows: {total:,}  ({time.monotonic()-started:.1f}s)", flush=True)
    # Deterministic subsample: same seed across reruns gives the same tile selection.
    if total > TARGET_TILE_COUNT:
        keep = np.random.default_rng(int(data["split_seed"])).choice(total, size=TARGET_TILE_COUNT, replace=False)
        keep.sort()
    else:
        keep = np.arange(total)
    rows = select_rows(sample_list, keep.tolist())
    # Sort by slide so each worker stays on one slide for many consecutive tiles.
    rows.sort(key=lambda r: r[0])
    args_iter = [(str(dataset_dir), *r) for r in rows]
    workers = int(os.environ.get("PREPROCESS_WORKERS", os.cpu_count() or 8))
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


if __name__ == "__main__":
    main()
