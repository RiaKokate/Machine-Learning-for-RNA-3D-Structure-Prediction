#!/usr/bin/env python
"""
Summarize all checkpoints in RNA_CHECKPOINT_DIR (or ./checkpoints): file size,
training metadata embedded in .pt files, and training curves from train_log.jsonl.

Usage:
  python scripts/checkpoint_metrics.py
  python scripts/checkpoint_metrics.py --ckpt-dir /path/to/checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch


def _root() -> Path:
    r = os.environ.get("RNA3D_ROOT")
    return Path(r).expanduser().resolve() if r else Path(__file__).resolve().parents[1]


def _summarize_state_dict(sd: dict) -> dict:
    n_tensors = 0
    n_params = 0
    for v in sd.values():
        if hasattr(v, "numel"):
            n_tensors += 1
            n_params += int(v.numel())
    return {"n_tensors": n_tensors, "n_params": n_params}


def inspect_pt(path: Path, device: str = "cpu") -> dict:
    out = {"file": str(path.name), "bytes": path.stat().st_size}
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        out["error"] = str(e)
        return out
    if isinstance(ck, dict):
        str_keys = [k for k in ck.keys() if isinstance(k, str)]
        out["keys"] = sorted(str_keys)
        if "epoch" in ck:
            out["epoch"] = ck["epoch"]
        meta = ck.get("meta")
        if isinstance(meta, dict):
            out["meta"] = {k: meta[k] for k in sorted(meta) if k in meta}
        m = ck.get("model")
        if m is None:
            m = ck.get("model_state_dict")
        if isinstance(m, dict):
            out.update(_summarize_state_dict(m))
        elif str_keys and all(isinstance(ck[k], torch.Tensor) for k in str_keys):
            out["kind"] = "raw_state_dict"
            out.update(_summarize_state_dict({k: ck[k] for k in str_keys}))
    else:
        out["note"] = f"unexpected type {type(ck)}"
    return out


def parse_train_log(path: Path) -> dict:
    if not path.is_file():
        return {"train_log": "missing"}
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {"train_log": str(path.name), "lines": 0}

    # Split into runs when epoch number drops (new training session)
    runs: list[list[dict]] = []
    cur: list[dict] = []
    prev_ep = -1
    for r in rows:
        ep = r.get("ep", 0)
        if cur and ep < prev_ep:
            runs.append(cur)
            cur = []
        cur.append(r)
        prev_ep = ep
    if cur:
        runs.append(cur)

    summary = {
        "train_log": str(path.name),
        "total_lines": len(rows),
        "n_runs": len(runs),
        "runs": [],
    }
    best_va_global = min((r.get("va_rmsd", float("inf")) for r in rows), default=float("inf"))
    summary["best_va_rmsd_overall"] = best_va_global

    for i, run in enumerate(runs, 1):
        va = [r.get("va_rmsd") for r in run if "va_rmsd" in r]
        tr = [r.get("tr_rmsd") for r in run if "tr_rmsd" in r]
        best_i = min(range(len(va)), key=lambda j: va[j]) if va else 0
        summary["runs"].append({
            "run_index": i,
            "epochs_logged": len(run),
            "ep_range": (run[0].get("ep"), run[-1].get("ep")),
            "best_va_rmsd": min(va) if va else None,
            "best_va_ep": run[best_i].get("ep") if va else None,
            "last_va_rmsd": va[-1] if va else None,
            "last_tr_rmsd": tr[-1] if tr else None,
        })
    return summary


def main():
    ap = argparse.ArgumentParser(description="Checkpoint and train_log summary")
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory with .pt files (default: RNA_CHECKPOINT_DIR or <repo>/checkpoints)",
    )
    args = ap.parse_args()
    root = _root()
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = Path(os.environ.get("RNA_CHECKPOINT_DIR", str(root / "checkpoints")))
    ckpt_dir = ckpt_dir.expanduser().resolve()

    print(f"Checkpoint directory: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        print("  (directory missing)")
        return

    pts = sorted(ckpt_dir.glob("*.pt"))
    print(f"\n=== PyTorch checkpoints ({len(pts)} files) ===\n")
    for p in pts:
        info = inspect_pt(p)
        print(json.dumps(info, indent=2))
        print()

    log_path = ckpt_dir / "train_log.jsonl"
    print("=== Training log ===\n")
    print(json.dumps(parse_train_log(log_path), indent=2))


if __name__ == "__main__":
    main()
