#!/usr/bin/env python
"""
Evaluate every .pt checkpoint that is compatible with RNAFoldModel in rna.py on the
same validation split (fold-0). Reports mean / median C4' RMSD (Å) and TM-score
(TM-style score as defined in rna.evaluate_sample).

Large third-party pretrained checkpoints use a different architecture and are skipped.

Usage (from repo root):
  python scripts/evaluate_checkpoints.py
  python scripts/evaluate_checkpoints.py --ckpt-dir ./checkpoints --max-chains 500

Requires RNA_DATASET_DIR (or data/dataset / data_with_pdb) with pdb/, msa/, list/ populated.
Loads only PDBs listed in list/valid_fold-{k} (default k=0), not the full training corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import rna  # noqa: E402


def _extract_student_state(ck: dict):
    if not isinstance(ck, dict):
        return None
    if "model" in ck and isinstance(ck["model"], dict):
        return ck["model"]
    if "model_state_dict" in ck and isinstance(ck["model_state_dict"], dict):
        return ck["model_state_dict"]
    return None


def _compatible_load(model: torch.nn.Module, sd: dict) -> tuple[bool, str]:
    if sd is None:
        return False, "no 'model' / 'model_state_dict' in checkpoint"
    ref = model.state_dict()
    ref_keys = set(ref.keys())
    sk = set(sd.keys())
    overlap = ref_keys & sk
    if len(overlap) < 0.85 * len(ref_keys):
        return False, f"param key overlap {len(overlap)}/{len(ref_keys)} (need different network)"
    for k in overlap:
        if ref[k].shape != sd[k].shape:
            return False, f"shape mismatch: {k} model={tuple(ref[k].shape)} ckpt={tuple(sd[k].shape)}"
    try:
        inc = model.load_state_dict(sd, strict=False)
        if len(inc.missing_keys) > max(8, int(0.12 * len(ref_keys))):
            return False, f"too many missing_keys ({len(inc.missing_keys)}) after load"
    except RuntimeError as e:
        return False, str(e)
    return True, ""


def _infer_pair_in_ch(sd: dict | None) -> int:
    """Match PairEncoder input width to checkpoint (45 = legacy, 46 = +MSA cov channel)."""
    if not sd:
        return 46

    def _pick(*names: str):
        for name in names:
            if name in sd:
                return sd[name]
            mod = f"module.{name}"
            if mod in sd:
                return sd[mod]
        return None

    # net.0 = LayerNorm(in_dim) -> bias shape (in_dim,)
    b0 = _pick("pair_enc.net.0.bias")
    if b0 is not None and getattr(b0, "ndim", 0) == 1:
        return int(b0.shape[0])

    # net.1 = Linear(in_dim, ...) -> weight (out, in_dim)
    w1 = _pick("pair_enc.net.1.weight")
    if w1 is not None and getattr(w1, "ndim", 0) == 2:
        return int(w1.shape[1])

    return 46


def _build_model(device: torch.device, pair_in_ch: int) -> torch.nn.Module:
    return rna.RNAFoldModel(
        d_model=rna.CFG["D_MODEL"],
        d_pair=rna.CFG["D_PAIR"],
        n_seq_layers=rna.CFG["N_SEQ_LAYERS"],
        n_pair_blocks=rna.CFG["N_PAIR_BLOCKS"],
        n_heads=rna.CFG["N_HEADS"],
        n_bins=rna.CFG["N_BINS"],
        use_grad_ckpt=rna.CFG["USE_GRAD_CKPT"],
        n_atoms=rna.N_ATOMS,
        max_len=rna.CFG["MAX_LEN"] + 16,
        pair_in_ch=pair_in_ch,
    ).to(device)


def main():
    ap = argparse.ArgumentParser(
        description="Mean/median C4' RMSD and TM-score for each student checkpoint in a folder."
    )
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="Folder with .pt files (default: RNA_CHECKPOINT_DIR or ./checkpoints)")
    ap.add_argument("--max-chains", type=int, default=200, help="Max validation chains to score (full val set may be larger)")
    ap.add_argument("--out-csv", type=Path, default=None, help="Summary table (default: EVAL_OUT/checkpoint_eval_summary.csv)")
    ap.add_argument("--verbose", action="store_true", help="Print per-chain lines from batch_evaluate")
    ap.add_argument("--val-fold", type=int, default=0, help="Use list/valid_fold-{k} (default: 0)")
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = Path(os.environ.get("RNA_CHECKPOINT_DIR", str(ROOT / "checkpoints")))
    ckpt_dir = ckpt_dir.expanduser().resolve()

    eval_out = Path(os.environ.get("RNA_EVAL_OUT", str(ROOT / "eval_out"))).expanduser().resolve()
    eval_out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_csv or (eval_out / "checkpoint_eval_summary.csv")

    device = rna.DEVICE
    print("Loading validation chains (PDBs for list/valid_fold only; no full-corpus scan) ...")
    val_ids = rna.read_validation_chain_ids(rna.LIST_DIR, val_fold=args.val_fold)
    if not val_ids:
        raise SystemExit(
            f"No ids in {rna.LIST_DIR / f'valid_fold-{args.val_fold}'}. "
            "Set RNA_DATASET_DIR to your dataset with pdb/msa/list."
        )
    val_s = rna.load_structure_samples_for_ids(
        rna.PDB_DIR, rna.MSA_DIR, val_ids, verbose=True
    )
    if not val_s:
        raise SystemExit("Validation PDBs could not be loaded (check id ↔ filename match).")

    n_eval = min(args.max_chains, len(val_s))
    print(f"Dataset: {rna.DATASET_ROOT}")
    print(f"Checkpoints: {ckpt_dir}")
    print(
        f"Validation chains (fold {args.val_fold}): {len(val_s):,} loaded | "
        f"scoring first {n_eval:,} on {device}\n"
    )

    rows_out = []
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt files in {ckpt_dir}")

    for path in pts:
        row = {
            "checkpoint": path.name,
            "status": "",
            "mean_rmsd": "",
            "median_rmsd": "",
            "mean_tm": "",
            "median_tm": "",
            "pct_rmsd_le_3A": "",
            "pct_rmsd_le_5A": "",
            "n_chains": "",
            "note": "",
        }
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = _extract_student_state(ck)
        pic = _infer_pair_in_ch(sd)
        model = _build_model(device, pair_in_ch=pic)
        ok, msg = _compatible_load(model, sd)
        if not ok:
            row["status"] = "skipped"
            row["note"] = msg
            print(f"--- {path.name} ---\n  SKIP: {msg}\n")
            rows_out.append(row)
            continue

        meta = ck.get("meta") if isinstance(ck, dict) else {}
        if isinstance(meta, dict) and "val_rmsd_A" in meta:
            row["note"] = f"train_meta_val_rmsd_A={meta['val_rmsd_A']:.4f}"

        df = rna.batch_evaluate(
            model, val_s, device, max_chains=args.max_chains, silent=not args.verbose
        )
        if df.empty:
            row["status"] = "error"
            row["note"] = "batch_evaluate returned empty"
            rows_out.append(row)
            continue

        row["status"] = "ok"
        row["mean_rmsd"] = f"{df.rmsd.mean():.4f}"
        row["median_rmsd"] = f"{df.rmsd.median():.4f}"
        row["mean_tm"] = f"{df.tm.mean():.4f}"
        row["median_tm"] = f"{df.tm.median():.4f}"
        row["pct_rmsd_le_3A"] = f"{(df.rmsd <= 3.0).mean() * 100:.2f}"
        row["pct_rmsd_le_5A"] = f"{(df.rmsd <= 5.0).mean() * 100:.2f}"
        row["n_chains"] = str(len(df))
        rows_out.append(row)

        per = eval_out / f"val_results__{path.stem}.csv"
        df.to_csv(per, index=False)

        print(f"--- {path.name} ---")
        print(f"  mean RMSD:   {df.rmsd.mean():.3f} A")
        print(f"  median RMSD: {df.rmsd.median():.3f} A")
        print(f"  mean TM:     {df.tm.mean():.4f}")
        print(f"  median TM:   {df.tm.median():.4f}")
        print(f"  % RMSD<=3A:  {row['pct_rmsd_le_3A']}%")
        print(f"  % RMSD<=5A:  {row['pct_rmsd_le_5A']}%")
        print(f"  chains:      {len(df)}")
        print(f"  per-chain:   {per}\n")

    import pandas as pd

    summary_df = pd.DataFrame(rows_out)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table: {summary_path}")

    ok_rows = [r for r in rows_out if r.get("status") == "ok"]
    if ok_rows:
        print("\n=== Comparable checkpoints (mean / median RMSD & TM) ===")
        for r in ok_rows:
            print(
                f"  {r['checkpoint']:<28}  "
                f"RMSD mean={r['mean_rmsd']} med={r['median_rmsd']} A | "
                f"TM mean={r['mean_tm']} med={r['median_tm']} | "
                f"<=3A={r['pct_rmsd_le_3A']}%  n={r['n_chains']}"
            )

    print("\nJSON summary:")
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
