# Auto-download for the downstream probe datasets in DATASET_ROOTS. The
# pretraining TCGA sample list is intentionally NOT auto-downloaded; we just
# warn if it's missing. URLs and unpack steps are vendored from
# /admin/home/paul/thunder/src/thunder/datasets/dataset/*.py so we don't take
# Thunder on as a runtime dependency.
#
# Usage:
#   python download_data.py            # download all missing probe datasets
#   python download_data.py pcam bach  # only the named ones

import gzip
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

from probe import CLASSIFICATION_DATASETS, DATASET_ROOTS, SEGMENTATION_DATASETS

TCGA_SAMPLE_LIST = Path("/block/TCGA/sample_dataset_30.txt")
ALL_DATASETS = CLASSIFICATION_DATASETS + SEGMENTATION_DATASETS


def _download(url, dst):
    print(f"  GET {url}\n   -> {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers={"User-Agent": "nanopath"}, timeout=60) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _unzip(src, dst_dir):
    with zipfile.ZipFile(src) as z:
        z.extractall(dst_dir)
    src.unlink()


def _ungzip(src, dst):
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    src.unlink()


def _untar(src, dst_dir):
    with tarfile.open(src) as t:
        t.extractall(dst_dir)
    src.unlink()


def download_pannuke(root):
    for fold in (1, 2, 3):
        zip_path = root / f"fold_{fold}.zip"
        _download(f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip", zip_path)
        _unzip(zip_path, root)
        (root / f"Fold {fold}").rename(root / f"Fold{fold}")


def download_pcam(root):
    base = "https://zenodo.org/api/records/2546921/files"
    for split in ("train", "valid", "test"):
        for kind in ("x", "y"):
            name = f"camelyonpatch_level_2_split_{split}_{kind}.h5"
            gz = root / (name + ".gz")
            _download(f"{base}/{name}.gz/content", gz)
            _ungzip(gz, root / name)


def download_bach(root):
    base = "https://zenodo.org/api/records/3632035/files"
    for name in ("ICIAR2018_BACH_Challenge.zip", "ICIAR2018_BACH_Challenge_TestDataset.zip"):
        zip_path = root / name
        _download(f"{base}/{name}/content", zip_path)
        _unzip(zip_path, root)


def download_bracs(root):
    cmd = ["wget", "--no-parent", "-nH", "-r", "--directory-prefix", str(root), "ftp://histoimage.na.icar.cnr.it/BRACS_RoI/"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def download_break_his(root):
    tar = root / "BreaKHis_v1.tar.gz"
    _download("http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz", tar)
    _untar(tar, root)


def download_mhist(root):
    print(
        "  mhist requires manual access. Visit https://bmirds.github.io/MHIST/#accessing-dataset,\n"
        "  fill in the form, then drop annotations.csv and images.zip into:\n"
        f"    {root}\n"
        "  After that, re-run `python download_data.py mhist` to unzip the images.",
        flush=True,
    )
    images_zip = root / "images.zip"
    if images_zip.exists():
        _unzip(images_zip, root)


DOWNLOADERS = {
    "pannuke": download_pannuke,
    "pcam": download_pcam,
    "bach": download_bach,
    "bracs": download_bracs,
    "break_his": download_break_his,
    "mhist": download_mhist,
}


def main():
    targets = sys.argv[1:] or ALL_DATASETS
    for dataset in targets:
        if dataset not in DOWNLOADERS:
            raise ValueError(f"unknown dataset: {dataset}; valid: {sorted(DOWNLOADERS)}")
    for dataset in targets:
        root = DATASET_ROOTS[dataset]
        if root.exists() and any(root.iterdir()):
            print(f"[skip] {dataset}: {root} already populated", flush=True)
            continue
        print(f"[fetch] {dataset} -> {root}", flush=True)
        root.mkdir(parents=True, exist_ok=True)
        DOWNLOADERS[dataset](root)
        print(f"[done] {dataset}", flush=True)
    if not TCGA_SAMPLE_LIST.exists():
        print(
            f"\n[warn] TCGA pretraining sample list missing at {TCGA_SAMPLE_LIST}.\n"
            "       This is intentionally NOT auto-downloaded. Obtain it from your\n"
            "       internal source and place it at the configured data.sample_list\n"
            "       path before running pretraining.",
            flush=True,
        )


if __name__ == "__main__":
    main()
