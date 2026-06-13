# Rebuild FINO metadata from TCGA clinical/cBioPortal CSVs plus patient-level RNA-seq.
# Default output is a sibling JSON so the committed artifact is never overwritten by accident.
# Usage:
#   python metadata/build_fino_meta.py rnaseq_dir=/data/hassan/tcga-rnaseq/fpkm_uq_nonzero_ge_50pct

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULTS = {
    "csv_dir": "metadata",
    "dataset_dir": "/data/nanopath_parquet",
    "rnaseq_dir": "",
    "rnaseq_csv": "",
    "gene_map_csv": "",
    "pathway_mask_csv": "",
    "out": "metadata/fino_meta.rebuilt.json",
    "max_patients": "0",
    "pca": "true",
}


def args():
    cfg = DEFAULTS.copy()
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        cfg[k] = os.path.expandvars(v)
    return cfg


def clean(s):
    s = s.dropna().astype(str).str.strip()
    return s[~s.str.lower().isin(("", "nan", "none"))]


def tile_barcodes(dataset_dir):
    return sorted({
        "-".join(p.split("/", 1)[0].split("-")[:3])
        for shard in sorted(Path(dataset_dir).glob("shard-*.parquet"))
        for p in pq.read_table(str(shard), columns=["path"], memory_map=True)["path"].to_pylist()
    })


def master_table(csv_dir, barcodes):
    clinical = pd.read_csv(Path(csv_dir) / "tcga_master_dataset.csv", low_memory=False).drop_duplicates("submitter_id").set_index("submitter_id")
    genomics = pd.read_csv(Path(csv_dir) / "tcga_master_cancer_genomics.csv", low_memory=False).drop_duplicates("submitter_id").set_index("submitter_id")
    df = clinical.combine_first(genomics)
    for col in genomics.columns:
        if col.startswith("cbio_"):
            df[col] = genomics[col]
    return df.reindex(barcodes)


def add_discrete(meta, name, s):
    s = clean(s)
    ids = {v: i for i, v in enumerate(sorted(s.unique()))}
    meta["discrete"][name] = {b: int(ids[v]) for b, v in s.items()}
    meta["n"][name] = len(ids)


def add_continuous(meta, name, s, log1p=False):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if log1p:
        s = np.log1p(s.clip(lower=0))
    z = (s - s.mean()) / (s.std() + 1e-8)
    meta["continuous"][name] = {b: round(float(v), 4) for b, v in z.items()}
    meta["cont_dim"][name] = 1


def add_base_factors(meta, df, barcodes):
    add_discrete(meta, "cancer", df["project_id"])
    add_discrete(meta, "tss", pd.Series({b: b.split("-")[1] for b in barcodes}))
    msi = pd.to_numeric(df["cbio_msi_score"], errors="coerce").dropna()
    meta["discrete"]["msi"] = {b: int(v) for b, v in (msi >= 0.4).items()}
    meta["n"]["msi"] = 2
    year = pd.to_numeric(df["year_of_diagnosis"], errors="coerce").dropna()
    add_discrete(meta, "year", pd.qcut(year.rank(method="first"), 5, labels=False).astype(str))
    subtype = pd.DataFrame({"project": clean(df["project_id"]).str.replace("TCGA-", "", regex=False), "subtype": clean(df["cbio_subtype"])})
    subtype = subtype.dropna()
    subtype = subtype[subtype["subtype"] != subtype["project"]]
    add_discrete(meta, "subtype", subtype["subtype"])
    grade = clean(df["tumor_grade"]).str.upper()
    grade = grade[grade.str.contains("G1|G2|LOW|G3|G4|HIGH", regex=True)]
    grade = grade.map(lambda x: "low" if ("G1" in x or "G2" in x or "LOW" in x) else "high")
    add_discrete(meta, "grade", grade)
    for name, col in (
        ("site", "primary_site"),
        ("organ", "tissue_or_organ_of_origin"),
        ("resection", "site_of_resection_or_biopsy"),
        ("tstage", "ajcc_pathologic_t"),
        ("nstage", "ajcc_pathologic_n"),
        ("stage", "ajcc_pathologic_stage"),
        ("sampletype", "sample_type"),
        ("section", "slide_section_location"),
        ("stageedition", "ajcc_staging_system_edition"),
        ("gender", "gender"),
        ("priortx", "prior_treatment"),
        ("morphology", "morphology"),
        ("diseasetype", "disease_type"),
        ("classif", "classification_of_tumor"),
    ):
        add_discrete(meta, name, df[col])
    for name, col in (("scanner", "scanner_id"), ("appmag", "appmag")):
        if col in df:
            add_discrete(meta, name, df[col])
    for name, col, logged in (
        ("necrosis", "slide_percent_necrosis", True),
        ("fga", "cbio_fraction_genome_altered", False),
        ("mutcount", "cbio_mutation_count", True),
        ("til", "slide_percent_lymphocyte_infiltration", True),
        ("age", "age_at_index", False),
        ("stromal", "slide_percent_stromal_cells", True),
        ("mpp", "mpp", False),
    ):
        if col in df:
            add_continuous(meta, name, df[col], logged)


def load_expression(cfg, df, barcodes):
    limit = int(cfg["max_patients"])
    if cfg["rnaseq_dir"]:
        root = Path(cfg["rnaseq_dir"])
        mp = pd.read_csv(root / "dx_slide_to_patient_target.csv").drop_duplicates("patient_id")
        mp = mp[mp["patient_id"].isin(barcodes)]
        if limit:
            mp = mp.head(limit)
        col = "patient_npy_abspath" if "patient_npy_abspath" in mp else "patient_npy_relpath"
        paths = [Path(p) if Path(str(p)).is_absolute() else root / str(p) for p in mp[col]]
        x = np.stack([np.load(p).astype("float32") for p in paths])
        genes = (root / "gene_ensembl_ids.txt").read_text().splitlines()
        return mp["patient_id"].tolist(), mp["cancer_type"].astype(str).tolist(), genes, x
    xdf = pd.read_csv(cfg["rnaseq_csv"], index_col=0, low_memory=False)
    xdf.index = xdf.index.astype(str).str[:12]
    xdf = xdf[~xdf.index.duplicated(keep="first")]
    xdf = xdf.reindex([b for b in barcodes if b in xdf.index])
    if limit:
        xdf = xdf.head(limit)
    x = xdf.to_numpy(dtype=np.float32)
    keep = (x > 0).mean(0) >= 0.5
    x = x[:, keep]
    genes = [g for g, k in zip(xdf.columns.astype(str), keep) if k]
    org = df.reindex(xdf.index)["project_id"].astype(str).str.replace("TCGA-", "", regex=False).tolist()
    return xdf.index.tolist(), org, genes, x


def store_vector(meta, name, bcs, z):
    meta["continuous"][name] = {bcs[i]: [round(float(x), 3) for x in z[i]] for i in range(len(bcs))}
    meta["cont_dim"][name] = int(z.shape[1])
    print(f"{name}: {z.shape[1]} dims, {len(bcs)} patients", flush=True)


def add_expr(meta, cfg, df, barcodes):
    bcs, org, genes, x = load_expression(cfg, df, barcodes)
    x = np.log1p(x)
    keep_rows = np.isfinite(x).all(1)
    x, org = x[keep_rows], np.asarray(org)[keep_rows]
    bcs = [b for b, keep in zip(bcs, keep_rows) if keep]
    print(f"rnaseq matrix: {x.shape[0]} patients x {x.shape[1]} genes", flush=True)
    orgs = [o for o in sorted(set(org)) if (org == o).sum() >= 10]
    within_var = sum(x[org == o].var(0) for o in orgs) / len(orgs)
    top512 = np.argsort(within_var)[::-1][:512]
    top256 = top512[:256]
    for name, idx in (("expr", top256), ("expr512", top512)):
        z = (x[:, idx] - x[:, idx].mean(0)) / (x[:, idx].std(0) + 1e-8)
        store_vector(meta, name, bcs, z)
    if cfg["pca"].lower() == "true":
        from sklearn.decomposition import PCA
        xs = (x - x.mean(0)) / (x.std(0) + 1e-8)
        z = PCA(n_components=128, svd_solver="randomized", random_state=0).fit_transform(xs)
        store_vector(meta, "expr_pca", bcs, (z - z.mean(0)) / (z.std(0) + 1e-8))
    if cfg["pathway_mask_csv"]:
        gene_map = pd.read_csv(cfg["gene_map_csv"], low_memory=False)
        sym_col = [c for c in gene_map.columns if "symbol" in c.lower()][0]
        ens_col = [c for c in gene_map.columns if "ensembl" in c.lower() or c.lower() in ("gene_id", "ensembl_id")][0]
        gene_map["_ens"] = gene_map[ens_col].astype(str).str.split(".").str[0]
        ens2sym = dict(zip(gene_map["_ens"], gene_map[sym_col].astype(str)))
        sym_idx = {}
        for j, gene in enumerate([g.split(";")[0].split(".")[0] for g in genes]):
            sym = ens2sym.get(gene, "")
            if sym:
                sym_idx.setdefault(sym, []).append(j)
        pm = pd.read_csv(cfg["pathway_mask_csv"], index_col=0)
        scores = []
        for pathway in pm.index:
            idx = [j for sym in pm.columns[pm.loc[pathway].values == 1] if sym in sym_idx for j in sym_idx[sym]]
            if len(idx) >= 5:
                scores.append(x[:, idx].mean(1))
        p = np.asarray(scores).T
        top = np.argsort(p.var(0))[::-1][:256]
        z = (p[:, top] - p[:, top].mean(0)) / (p[:, top].std(0) + 1e-8)
        store_vector(meta, "expr_path", bcs, z)


def main():
    cfg = args()
    barcodes = tile_barcodes(cfg["dataset_dir"])
    df = master_table(cfg["csv_dir"], barcodes)
    meta = {"discrete": {}, "continuous": {}, "n": {}, "cont_dim": {}}
    add_base_factors(meta, df, barcodes)
    add_expr(meta, cfg, df, barcodes)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {out}: {len(meta['discrete'])} discrete, {len(meta['continuous'])} continuous", flush=True)


if __name__ == "__main__":
    main()
