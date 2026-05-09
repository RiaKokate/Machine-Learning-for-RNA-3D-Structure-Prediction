#!/usr/bin/env python
"""
Exploratory summaries for RNA 3D project data: standard layout (pdb/msa/list),
optional CSV sequence/label files, and MSA directory statistics.

Usage:
  python scripts/eda_rna_data.py
  python scripts/eda_rna_data.py --data-root /path/to/data
"""
from __future__ import annotations

import argparse
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _root() -> Path:
    r = os.environ.get("RNA3D_ROOT")
    return Path(r).expanduser().resolve() if r else Path(__file__).resolve().parents[1]


def seq_stats(sequences: pd.Series, name: str, max_sample: int = 200_000) -> None:
    s = sequences.dropna().astype(str)
    if len(s) > max_sample:
        s = s.sample(max_sample, random_state=42)
    lens = s.str.len()
    print(f"\n--- {name} (n={len(s):,} sampled rows) ---")
    print(f"  length: min={lens.min()}  median={lens.median():.0f}  max={lens.max()}")
    print(f"  mean length: {lens.mean():.1f}")
    all_bases = "".join(s.head(50_000).tolist())
    cnt = Counter(c for c in all_bases.upper() if c in "AUGC")
    tot = sum(cnt.values()) or 1
    print("  base freq (sample): " + " ".join(f"{b}:{cnt.get(b,0)/tot:.3f}" for b in "AUGC"))


def a3m_quick_stats(path: Path, max_lines: int = 400) -> dict:
    n_seq = 0
    width = None
    with path.open(errors="replace") as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            if line.startswith(">"):
                n_seq += 1
            elif line.strip() and width is None and not line.startswith("#"):
                width = len(line.strip())
    return {"n_seq_head": n_seq, "cols_guess": width}


def eda_dataset_tree(root: Path) -> None:
    print(f"\n=== Dataset tree: {root} ===")
    for sub in ("pdb", "msa", "list"):
        p = root / sub
        print(f"  {sub}/  exists={p.is_dir()}")
        if p.is_dir():
            files = list(p.iterdir())
            print(f"        files: {len(files):,}")


def eda_msa_dir(msa: Path, sample: int = 500) -> None:
    if not msa.is_dir():
        return
    files = [f for f in msa.iterdir() if f.suffix.lower() in (".a3m", ".sto", ".sto1")]
    print(f"\n=== MSA directory: {msa} ===")
    print(f"  alignments (.a3m/.sto): {len(files):,}")
    if not files:
        return
    sizes = np.array([f.stat().st_size for f in files])
    print(f"  file size (bytes): min={sizes.min()}  median={np.median(sizes):.0f}  max={sizes.max()}")
    rng = np.random.default_rng(42)
    pick = list(rng.choice(files, size=min(sample, len(files)), replace=False))
    seq_counts = []
    for f in pick[:20]:
        st = a3m_quick_stats(f)
        seq_counts.append(st.get("n_seq_head") or 0)
    if seq_counts:
        print(f"  sample (20 files): mean '>' lines in first 400 lines: {np.mean(seq_counts):.1f}")


def eda_csv(root: Path) -> None:
    candidates = [
        "train_seqs_combined.csv",
        "primary_secondary (1).csv",
        "train_labels_combined.csv",
        "train_labels_fullatom.csv",
    ]
    for name in candidates:
        p = root / name
        if not p.is_file():
            continue
        print(f"\n=== CSV: {name} ({p.stat().st_size/1e6:.1f} MB) ===")
        try:
            df = pd.read_csv(p, nrows=5000)
        except Exception as e:
            print(f"  skip read: {e}")
            continue
        print(f"  columns: {list(df.columns)}")
        if "sequence" in df.columns:
            seq_stats(df["sequence"], name)
        sz = p.stat().st_size
        if sz < 80_000_000:
            with p.open(errors="replace") as f:
                nlines = sum(1 for _ in f) - 1
            print(f"  rows (excl. header): {max(nlines, 0):,}")
        else:
            print(f"  rows: (file {sz/1e9:.2f} GB - skipped full line count)")
        if "secondary_structure" in df.columns:
            ss = df["secondary_structure"].dropna().astype(str)
            if len(ss):
                dots = (ss.str.count(r"\.")).mean()
                pairs = (ss.str.count(r"[\(\)]")).mean()
                print(f"  secondary_structure (sample): mean '.' {dots:.1f}  bracket chars {pairs:.1f}")


def eda_zips(root: Path) -> None:
    for z in root.glob("*.zip"):
        print(f"\n=== Zip: {z.name} ===")
        try:
            with zipfile.ZipFile(z, "r") as zf:
                names = zf.namelist()
                print(f"  entries: {len(names):,}")
                print(f"  example: {names[:5]}")
        except zipfile.BadZipFile:
            print("  (invalid zip)")


def _default_dataset_eda(root_proj: Path) -> Path:
    r1 = root_proj / "data" / "dataset"
    r2 = root_proj / "data_with_pdb"

    def has_pdbs(root: Path) -> bool:
        d = root / "pdb"
        if not d.is_dir():
            return False
        try:
            return any(d.glob("*.pdb"))
        except OSError:
            return False

    env = os.environ.get("RNA_DATASET_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if has_pdbs(p):
            return p
        print(f"[WARN] RNA_DATASET_DIR={p} has no pdb/*.pdb; using data/dataset or data_with_pdb.")

    if has_pdbs(r1):
        return r1.resolve()
    if has_pdbs(r2):
        return r2.resolve()
    return (Path(env).expanduser().resolve() if env else r1.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Default: RNA_DATASET_DIR or data/dataset / data_with_pdb",
    )
    args = ap.parse_args()
    root_proj = _root()
    data_root = args.data_root
    if data_root is None:
        data_root = _default_dataset_eda(root_proj)
    data_root = data_root.expanduser().resolve()
    parent = data_root.parent if data_root.name == "dataset" else data_root

    print(f"Project root: {root_proj}")
    print(f"Dataset root (RNA_DATASET_DIR): {data_root}")

    eda_dataset_tree(data_root)
    if (parent / "msa").is_dir() and parent / "msa" != data_root / "msa":
        eda_msa_dir(parent / "msa")
    eda_msa_dir(data_root / "msa")
    eda_csv(root_proj / "data")
    eda_zips(root_proj / "data")


if __name__ == "__main__":
    main()
