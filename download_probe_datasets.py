# Auto-download for the downstream probe datasets in DATASET_ROOTS. TCGA
# pretraining data is handled separately by download_TCGA.sh because it is a
# multi-TB WSI download. URLs and unpack steps are vendored from
# https://github.com/MedARC-AI/thunder/tree/main/src/thunder/datasets/dataset/*.py
# so we don't take Thunder on as a runtime dependency.
#
# Usage:
#   python download_probe_datasets.py            # download all missing probe datasets
#   python download_probe_datasets.py pcam bach  # only the named ones

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import requests

from probe import DATASET_ROOTS

TCGA_SAMPLE_LIST = Path("/block/TCGA/sample_dataset_30.txt")


# Stream a URL to disk in chunks so large probe archives do not sit in memory.
def download(url, dst):
    print(f"  GET {url}\n   -> {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers={"User-Agent": "nanopath"}, timeout=60) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


# Fetch and unpack the three PanNuke folds used by the segmentation probe.
def download_pannuke(root):
    for fold in (1, 2, 3):
        zip_path = root / f"fold_{fold}.zip"
        download(f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()
        (root / f"Fold {fold}").rename(root / f"Fold{fold}")


# Fetch PatchCamelyon h5 splits; probe.py later samples a fixed subset.
def download_pcam(root):
    base = "https://zenodo.org/api/records/2546921/files"
    for split in ("train", "valid", "test"):
        for kind in ("x", "y"):
            name = f"camelyonpatch_level_2_split_{split}_{kind}.h5"
            gz = root / (name + ".gz")
            download(f"{base}/{name}.gz/content", gz)
            with gzip.open(gz, "rb") as fin, (root / name).open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            gz.unlink()


# Fetch the BACH train and test archives used by the classification probe splits.
def download_bach(root):
    base = "https://zenodo.org/api/records/3632035/files"
    for name in ("ICIAR2018_BACH_Challenge.zip", "ICIAR2018_BACH_Challenge_TestDataset.zip"):
        zip_path = root / name
        download(f"{base}/{name}/content", zip_path)
        shutil.unpack_archive(zip_path, root)
        zip_path.unlink()


# Mirror BRACS ROI files via wget because the source is exposed as FTP.
def download_bracs(root):
    cmd = ["wget", "--no-parent", "-nH", "-r", "--directory-prefix", str(root), "ftp://histoimage.na.icar.cnr.it/BRACS_RoI/"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# Fetch and unpack BreaKHis images for the classification probe.
def download_break_his(root):
    tar = root / "BreaKHis_v1.tar.gz"
    download("http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz", tar)
    shutil.unpack_archive(tar, root)
    tar.unlink()


# MHIST requires manual form access, so this function only unpacks local files when present.
def download_mhist(root):
    print(
        "  mhist requires manual access. Visit https://bmirds.github.io/MHIST/#accessing-dataset,\n"
        "  fill in the form, then drop annotations.csv and images.zip into:\n"
        f"    {root}\n"
        "  After that, re-run `python download_probe_datasets.py mhist` to unzip the images.",
        flush=True,
    )
    images_zip = root / "images.zip"
    if images_zip.exists():
        shutil.unpack_archive(images_zip, root)
        images_zip.unlink()


DOWNLOADERS = {
    "bach": download_bach,
    "bracs": download_bracs,
    "break_his": download_break_his,
    "mhist": download_mhist,
    "pcam": download_pcam,
    "pannuke": download_pannuke,
}


# CLI entry point: download all missing probe datasets or only names passed on the command line.
def main():
    targets = sys.argv[1:] or DOWNLOADERS
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
            "       Probe datasets are handled here, but TCGA pretraining data is not.\n"
            "       Follow README.md and run `bash download_TCGA.sh /data/TCGA 8`,\n"
            "       then point data.sample_list at /data/TCGA/sample_dataset_30.txt\n"
            "       or place the file at the configured path before pretraining.",
            flush=True,
        )


if __name__ == "__main__":
    main()
