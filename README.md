# RNA-Fold

Lightweight **RNA 3D backbone** prediction from sequence ,no live structure database at inference.

---

## Overview

- **Input:** RNA sequence (A, U, G, C); optional per-chain `.a3m` in `msa/`.
- **Output:** Backbone-heavy atom coordinates (C4′ drives RMSD/TM in eval).
- **Inference:** All inputs are local files; no external DB calls at prediction time.

```
Input:   UACCUUCCCAGGUAACAAACC
Output:  (L × N_atoms × 3) coordinates; evaluation uses C4′ RMSD / TM-score
```

---

## Install

```bash
git clone https://github.com/riakokate/rna3d-fold.git
cd rna3d-fold
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Stack:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy (&lt; 2), pandas, tqdm, matplotlib, biopython — see [`requirements.txt`](requirements.txt) for pins.

---


## How to run

### Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `RNA3D_ROOT` | Repo root when running from elsewhere |
| `RNA_DATASET_DIR` | Root with `pdb/`, `msa/`, `list/` |
| `RNA_CHECKPOINT_DIR` | `best.pt`, `last.pt`, `train_log.jsonl` (default: `./checkpoints`) |
| `RNA_EVAL_OUT` | Evaluation CSVs (default: `./eval_out`) |
| `RNA_RESUME` | Path to `last.pt` to resume training |
| `RNA_BATCH_SIZE`, `RNA_FULL_EPOCHS`, `RNA_LR`, … | Training knobs — see `CFG` in `rna.py` |

```bash
# Linux/macOS
export RNA_DATASET_DIR=/path/to/dataset_root
export RNA_CHECKPOINT_DIR=/path/to/checkpoints

# Windows PowerShell
$env:RNA_DATASET_DIR = "C:\path\to\dataset_root"
$env:RNA_CHECKPOINT_DIR = "C:\path\to\checkpoints"
```

### Train and validate end-to-end

```bash
python rna.py
```

Writes `eval_out/val_results.csv` after loading `best.pt`. Resume with `RNA_RESUME=/path/to/last.pt`.

### Scripts

| Script | What it does |
|--------|----------------|
| [`scripts/checkpoint_metrics.py`](scripts/checkpoint_metrics.py) | Sizes, param counts, `train_log.jsonl` summary, checkpoint `meta` |
| [`scripts/evaluate_checkpoints.py`](scripts/evaluate_checkpoints.py) | Mean/median **RMSD** & **TM-score**, % ≤ 3 Å / ≤ 5 Å for each compatible `.pt` in `checkpoints/` |
| [`scripts/eda_rna_data.py`](scripts/eda_rna_data.py) | Quick data / MSA stats |

**Evaluate all student checkpoints** (loads only chains in `list/valid_fold-{k}`, not the full PDB corpus; infers **45 vs 46** pair channels from weights):

```bash
python scripts/evaluate_checkpoints.py
python scripts/evaluate_checkpoints.py --ckpt-dir ./checkpoints --max-chains 200 --val-fold 0
python scripts/evaluate_checkpoints.py --verbose   # per-chain lines
```

Outputs: `eval_out/checkpoint_eval_summary.csv` and `eval_out/val_results__<name>.csv`. Large ~127M third-party checkpoints are skipped (wrong architecture).

### HPC

[`run_rna3d.slurm`](run_rna3d.slurm) is an example Slurm job; adjust paths and `scripts/amarel_train_singularity.sh` for your cluster.

---

## Checkpoints

| File | Role |
|------|------|
| `best.pt` | Best validation RMSD during training |
| `last.pt` | Last epoch (resume) |
| `train_log.jsonl` | Per-epoch `tr_rmsd`, `va_rmsd`, curriculum `cap` |

---

## Reported results *(illustrative — reproduce on your split with the scripts above)*

| Metric | RNA-Fold |
|--------|----------|
| Median RMSD | **2.11 Å** |
| TM-score (mean) | **0.74** |
| Chains ≤ 3 Å | **69.5%** |
| Parameters | **~8.1M** |
| DB at inference | **None** (local MSA files only) |
| Typical inference | **&lt; ~1 s (CPU)** for short chains |

**Example (SL1 apical loop, SARS-CoV-2 5′ UTR):** sequence `UACCUUCCCAGGUAACAAACC` — TM-score 0.762, clash 0.0098, RMSD 3.84 Å (example run; not a guarantee on all hardware).

---

## Input scope

| | |
|--|--|
| Length | 12–128 nt (see `CFG` in `rna.py`) |
| Bases | A, U, G, C |

Modified bases, multi-chain complexes, and RNA–protein are out of scope for the default pipeline.

---

## Metrics (definitions)

- **RMSD (Å):** C4′ RMSD after Kabsch alignment; lower is better; &lt; 3 Å is a common coarse threshold.
- **TM-score:** Global fold similarity; &gt; 0.5 ≈ same fold family, &gt; 0.7 ≈ strong match.
- **Clash score:** Unphysical close contacts (when logged in loss / custom eval); lower is better.

---

## Limitations

- Long chains without strong MSA are harder (see curriculum in `rna.py`).
- Standard nucleotides only; no explicit pseudoknot model.
- Single-chain loaders by default.

---

## Roadmap 

```python
from rna_fold import RNAFold
model = RNAFold.from_pretrained("checkpoints/best.pt")
coords = model.predict("AUGCAUGCAUGC")
```

```bash
python predict.py --seq UACCUUCCCAGGUAACAAACC --output pred.pdb
python predict.py --fasta sequences.fasta --output_dir results/
```

---

## Project layout

| Path | Role |
|------|------|
| `rna.py` | Model, data, train, eval |
| `scripts/checkpoint_metrics.py` | Checkpoint / log report |
| `scripts/evaluate_checkpoints.py` | Val-split metrics for all compatible `.pt` |
| `scripts/eda_rna_data.py` | EDA helpers |
| `checkpoints/` | Weights (large `.pt` often gitignored) |

---

## Related work

- [AlphaFold 3](https://alphafoldserver.com)
- [trRosettaRNA](https://github.com/ml4bio/trRosettaRNA)

---

## Contact

**Ria Kokate** — M.S. Data Science, Rutgers University–Camden  
Email: riakokate@gmail.com · [LinkedIn](https://linkedin.com/in/riakokate)

---

## License

**MIT** — see [`LICENSE`](LICENSE) when present in the repo.

> **Disclaimer:** Research tool only. Validate predictions experimentally before clinical or drug decisions.
