#!/usr/bin/env python
# coding: utf-8
"""
rna.py  
═══════════════════════════════════════════════════════════════════════════════
RNA 3D structure prediction — sequence + MSA → atomic coordinates
Target: <3 Å C4' RMSD (short chains), <10 Å (long chains)

"""
from __future__ import annotations
import os, json, math, random, csv, hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from torch.amp import GradScaler, autocast; _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast; _AMP_NEW = False

# ─── §0  Device ───────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)
if DEVICE.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} | mem: {props.total_memory/1e9:.1f} GB')
    _BF16   = torch.cuda.is_bf16_supported()
    _ADTYPE = torch.bfloat16 if _BF16 else torch.float16
    print(f'AMP dtype: {"bfloat16" if _BF16 else "float16"}')
else:
    _BF16 = False; _ADTYPE = torch.float32

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ─── §1  Paths & config ───────────────────────────────────────────────────────
def _base():
    e = os.environ.get('RNA3D_ROOT')
    return Path(e).expanduser().resolve() if e else Path(__file__).resolve().parent

BASE = _base()


def _default_dataset_root() -> Path:
    """
    Canonical layout: <repo>/data/dataset/{pdb,msa,list}.
    If that tree has no PDBs, use <repo>/data_with_pdb (same layout).
    Override with RNA_DATASET_DIR (invalid path falls back with a warning).
    """
    r1 = BASE / 'data' / 'dataset'
    r2 = BASE / 'data_with_pdb'

    def _has_pdbs(root: Path) -> bool:
        d = root / 'pdb'
        if not d.is_dir():
            return False
        try:
            return any(d.glob('*.pdb'))
        except OSError:
            return False

    env = os.environ.get('RNA_DATASET_DIR')
    if env:
        p = Path(env).expanduser().resolve()
        if _has_pdbs(p):
            return p
        print(f'[WARN] RNA_DATASET_DIR={p} has no usable pdb/*.pdb; trying data/dataset then data_with_pdb.')

    if _has_pdbs(r1):
        return r1.resolve()
    if _has_pdbs(r2):
        return r2.resolve()
    return (Path(env).expanduser().resolve() if env else r1.resolve())


DATASET_ROOT = _default_dataset_root()
PDB_DIR      = DATASET_ROOT / 'pdb'
MSA_DIR      = DATASET_ROOT / 'msa'
LIST_DIR     = DATASET_ROOT / 'list'

EVAL_OUT  = Path(os.environ.get('RNA_EVAL_OUT',       str(BASE / 'eval_out'))).expanduser()
CKPT_DIR  = Path(os.environ.get('RNA_CHECKPOINT_DIR', str(BASE / 'checkpoints'))).expanduser()
for _d in [EVAL_OUT, CKPT_DIR]: _d.mkdir(parents=True, exist_ok=True)

print(f'Dataset PDB dir : {PDB_DIR}  {"OK" if PDB_DIR.exists() else "MISSING"}')
print(f'Dataset MSA dir : {MSA_DIR}  {"OK" if MSA_DIR.exists() else "MISSING"}')
print(f'Dataset list dir: {LIST_DIR}  {"OK" if LIST_DIR.exists() else "MISSING"}')

TOK      = {'PAD':0,'A':1,'U':2,'G':3,'C':4,'-':5}
ALPHABET = {'A','U','G','C'}
V        = len(TOK)

# Atom names in training PDB files (backbone-heavy set)
ATOM_NAMES = ["C4'","P","C3'","O3'","O5'","C5'","O4'","C1'","N"]
N_ATOMS    = len(ATOM_NAMES)

CFG = dict(
    MAX_LEN          = 128,
    MIN_CHAIN_LEN    = 12,
    SPLIT_SEED       = 42,
    # Model
    D_MODEL          = 256,
    D_PAIR           = 128,
    N_SEQ_LAYERS     = 8,
    N_PAIR_BLOCKS    = 4,
    N_HEADS          = 8,
    N_BINS           = 36,
    N_RECYCLES       = int(os.environ.get('RNA_N_RECYCLES',   '3')),
    USE_GRAD_CKPT    = (os.environ.get('RNA_USE_GRAD_CKPT',  '0') == '1'),
    # Training
    BATCH_SIZE       = int(os.environ.get('RNA_BATCH_SIZE',   '4')),
    FULL_EPOCHS      = int(os.environ.get('RNA_FULL_EPOCHS', '100')),
    LR               = float(os.environ.get('RNA_LR',         '3e-4')),
    LR_WARMUP_STEPS  = 500,
    LR_ETA_MIN       = 5e-6,
    WEIGHT_DECAY     = 1e-4,
    GRAD_CLIP        = float(os.environ.get('RNA_GRAD_CLIP',  '0.5')),
    GRAD_ACCUM       = int(os.environ.get('RNA_GRAD_ACCUM',   '4')),
    USE_AMP          = (os.environ.get('RNA_USE_AMP',         '1') == '1'),
    NUM_WORKERS      = 0,
    # Loss weights
    W_RMSD           = float(os.environ.get('RNA_W_RMSD',        '10.0')),
    W_BOND           = float(os.environ.get('RNA_W_BOND',        '5.0')),
    W_LOCAL_DIST     = float(os.environ.get('RNA_W_LOCAL_DIST',  '10.0')),
    W_BB_ANGLE       = float(os.environ.get('RNA_W_BB_ANGLE',    '3.0')),
    W_BB_DIHEDRAL    = float(os.environ.get('RNA_W_BB_DIHEDRAL', '1.5')),
    W_PAIR_DIST      = float(os.environ.get('RNA_W_PAIR_DIST',   '10.0')),
    W_SS_AUX         = float(os.environ.get('RNA_W_SS_AUX',      '1.0')),
    W_ATOM_DIST      = float(os.environ.get('RNA_W_ATOM_DIST',   '2.0')),
    W_DIST_NLL       = float(os.environ.get('RNA_W_DIST_NLL',    '0.5')),
    W_RG             = float(os.environ.get('RNA_W_RG',          '2.0')),
    W_CLASH          = float(os.environ.get('RNA_W_CLASH',       '1.0')),
    W_MSA_CONS       = float(os.environ.get('RNA_W_MSA_CONS',    '0.5')),
    # Curriculum  (ep 1-10: L≤32, 11-30: L≤64, 31-60: L≤96, 61-100: L≤128)
    CURR_P1_EPOCHS   = 10,
    CURR_P2_EPOCHS   = 30,
    CURR_P3_EPOCHS   = 60,
    CURR_P4_EPOCHS   = 100,
    PHASE2_START     = 11,
    GEOM_FULL_EPOCH  = 20,
    # Geometry prior
    IDEAL_C4_BOND    = 6.1,
    IDEAL_TWIST_DEG  = 33.0,
    IDEAL_RISE_ANG   = 2.8,
)
print(f'CFG D_MODEL={CFG["D_MODEL"]} D_PAIR={CFG["D_PAIR"]} '
      f'RECYCLES={CFG["N_RECYCLES"]} BATCH={CFG["BATCH_SIZE"]} '
      f'AMP={CFG["USE_AMP"]} EPOCHS={CFG["FULL_EPOCHS"]}')
print('FILE_VERSION=v12-msa')

# ─── §2  Data: parse PDB + A3M ────────────────────────────────────────────────
@dataclass
class RNASample:
    seq:       str
    coords:    np.ndarray          # (L, N_ATOMS, 3)
    chain_id:  str | None = None
    atom_mask: np.ndarray | None = None   # (L, N_ATOMS) bool
    msa_cons:  np.ndarray | None = None   # (L, 4) conservation
    msa_cov:   np.ndarray | None = None   # (L, L) mutual information
    has_msa:   bool = False
    atom_names: list = field(default_factory=lambda: ATOM_NAMES)

    @property
    def n_atoms(self): return self.coords.shape[1] if self.coords.ndim==3 else 1


# ── PDB parser ────────────────────────────────────────────────────────────────
def parse_backbone_pdb(pdb_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Parse a training-format PDB file (per-residue backbone atoms).
    Returns (sequence, coords (L,9,3), atom_mask (L,9)).
    Atom order: C4' P C3' O3' O5' C5' O4' C1' N
    """
    RES3 = {'ADE':'A','URA':'U','GUA':'G','CYT':'C',
             'A':'A','U':'U','G':'G','C':'C',
             'DA':'A','DT':'U','DG':'G','DC':'C'}  # fallback
    WANT = {"C4'":0,"P":1,"C3'":2,"O3'":3,"O5'":4,"C5'":5,"O4'":6,"C1'":7,"N":8}
    # These PDBs use single-letter residue names already
    residues: dict[int, dict] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            atom_name = line[12:16].strip()
            res_name  = line[17:20].strip()
            try: res_id = int(line[22:26])
            except: continue
            if atom_name not in WANT: continue
            try:
                x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
            except: continue
            base = RES3.get(res_name, res_name[0] if res_name else 'N')
            if base not in ALPHABET: continue
            if res_id not in residues:
                residues[res_id] = {'base': base, 'atoms': {}}
            residues[res_id]['atoms'][atom_name] = (x,y,z)

    if not residues:
        return '', np.zeros((0,9,3)), np.zeros((0,9), dtype=bool)

    sorted_res = sorted(residues.items())
    seq = ''.join(r['base'] for _,r in sorted_res)
    L   = len(sorted_res)
    coords    = np.full((L,9,3), np.nan, dtype=np.float32)
    atom_mask = np.zeros((L,9), dtype=bool)
    for i,(_,r) in enumerate(sorted_res):
        for aname, aidx in WANT.items():
            if aname in r['atoms']:
                coords[i,aidx] = r['atoms'][aname]
                atom_mask[i,aidx] = True
    return seq, coords, atom_mask


# ── A3M parser ────────────────────────────────────────────────────────────────
_MSA_VOCAB = {'A':0,'U':1,'G':2,'C':3}

def parse_a3m(path: Path, L_target: int, max_seqs: int = 512
              ) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Parse a .a3m MSA file → conservation (L,4) and covariance (L,L).
    Returns (conservation, covariance, has_real_msa).
    Falls back to zeros if file is absent or single-sequence.
    """
    empty_cons = np.zeros((L_target, 4), dtype=np.float32)
    empty_cov  = np.zeros((L_target, L_target), dtype=np.float32)

    if not path.is_file() or path.stat().st_size < 200:
        return empty_cons, empty_cov, False

    # Parse sequences from a3m
    seqs, cur_name, cur_seq = [], None, []
    try:
        with open(path, errors='replace') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('>'):
                    if cur_seq: seqs.append(''.join(cur_seq))
                    cur_name = line[1:]
                    cur_seq  = []
                else:
                    cur_seq.append(line)
            if cur_seq: seqs.append(''.join(cur_seq))
    except Exception:
        return empty_cons, empty_cov, False

    if len(seqs) < 2:
        return empty_cons, empty_cov, False

    # Query sequence is first; remove insertion columns (lowercase in a3m)
    query = seqs[0]
    # Column mask: positions that are NOT insertions in the query
    col_mask = [not c.islower() for c in query]
    query_clean = ''.join(c.upper() for c, keep in zip(query, col_mask) if keep)
    L_q = len(query_clean)

    # Process each sequence — keep only aligned columns
    oh_list = []
    for seq in seqs[1:max_seqs]:
        aln = []
        j = 0
        for c, keep in zip(seq, col_mask):
            if keep:
                aln.append(c.upper())
                j += 1
        if len(aln) != L_q:
            continue
        oh = np.zeros((L_q, 5), dtype=np.float32)  # 4 bases + gap
        for k, c in enumerate(aln):
            if c in _MSA_VOCAB: oh[k, _MSA_VOCAB[c]] = 1.0
            else:               oh[k, 4] = 1.0
        oh_list.append(oh)

    if not oh_list:
        return empty_cons, empty_cov, False

    oh_arr = np.stack(oh_list, axis=0)  # (N, L_q, 5)

    # Conservation: frequency per column (L_q, 4)
    cons_q = oh_arr[:,:,:4].mean(0).astype(np.float32)

    # Covariance via mutual information (L_q, L_q) — capped at max_L for speed
    max_L = min(L_q, L_target, 128)
    cov_q = np.zeros((max_L, max_L), dtype=np.float32)
    eps   = 1e-10
    f_i   = cons_q[:max_L] + eps  # (max_L, 4)
    oh_b  = oh_arr[:, :max_L, :4]  # (N, max_L, 4)
    N     = oh_b.shape[0]
    for i in range(max_L):
        for j in range(i+1, max_L):
            fij = (oh_b[:,i,:,None] * oh_b[:,j,None,:]).mean(0)  # (4,4)
            out = f_i[i,:,None] * f_i[j,None,:]
            mask = (fij > eps) & (out > eps)
            if mask.any():
                mi = float((fij[mask] * np.log(fij[mask] / out[mask])).sum())
                cov_q[i,j] = cov_q[j,i] = mi
    mx = cov_q.max()
    if mx > 0: cov_q /= mx

    # Pad/crop to L_target
    def _pad2d(arr, L):
        h, w = arr.shape
        out = np.zeros((L, L), dtype=np.float32)
        h2 = min(h, L); w2 = min(w, L)
        out[:h2,:w2] = arr[:h2,:w2]
        return out
    def _pad1d(arr, L):
        L2 = min(len(arr), L)
        out = np.zeros((L, arr.shape[1]), dtype=np.float32)
        out[:L2] = arr[:L2]
        return out

    cons_out = _pad1d(cons_q, L_target)
    cov_out  = _pad2d(cov_q,  L_target)
    return cons_out, cov_out, True


# ── Load structure samples ─────────────────────────────────────────────────────
def load_structure_samples(pdb_dir: Path, msa_dir: Path,
                          max_len: int = 128, min_len: int = 12,
                          verbose: bool = True) -> list[RNASample]:
    """Load all chains from pdb_dir, attach MSA features where available."""
    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    if not pdb_files:
        raise FileNotFoundError(f'No .pdb files found in {pdb_dir}')
    if verbose:
        print(f'Found {len(pdb_files):,} PDB files in {pdb_dir}')
        print(f'MSA dir: {msa_dir}  ({len(list(msa_dir.glob("*.a3m"))):,} .a3m files)')

    samples, skipped = [], 0
    for pdb_path in tqdm(pdb_files, desc='Loading PDBs', leave=False, disable=not verbose):
        cid = pdb_path.stem  # e.g. "1a34_A"
        try:
            seq, coords, atom_mask = parse_backbone_pdb(pdb_path)
        except Exception as e:
            skipped += 1; continue

        L = len(seq)
        if L < min_len or L > max_len:
            skipped += 1; continue
        if not np.isfinite(coords[:,0,:]).all():  # C4' must be finite
            skipped += 1; continue

        # MSA features
        a3m_path = msa_dir / f'{cid}.a3m'
        cons, cov, has_msa = parse_a3m(a3m_path, L_target=L)

        samples.append(RNASample(
            seq=seq, coords=coords, chain_id=cid.upper(),
            atom_mask=atom_mask, msa_cons=cons, msa_cov=cov,
            has_msa=has_msa, atom_names=ATOM_NAMES))

    if verbose:
        n_msa = sum(1 for s in samples if s.has_msa)
        print(f'Loaded: {len(samples):,} chains  |  skipped: {skipped:,}')
        print(f'With real MSA: {n_msa:,} ({n_msa/max(len(samples),1)*100:.0f}%)')
    return samples


def read_validation_chain_ids(list_dir: Path, val_fold: int = 0) -> set[str]:
    """IDs listed in list/valid_fold-{k} (one per line)."""
    val_file = list_dir / f'valid_fold-{val_fold}'
    if not val_file.is_file():
        return set()
    return {ln.strip() for ln in val_file.read_text().splitlines() if ln.strip()}


def load_structure_samples_for_ids(
    pdb_dir: Path,
    msa_dir: Path,
    chain_ids: set[str],
    max_len: int = 128,
    min_len: int = 12,
    verbose: bool = True,
) -> list[RNASample]:
    """
    Load only PDBs whose chain id appears in chain_ids (case/spacing insensitive).
    Faster than load_structure_samples when evaluating a fixed val list.
    """
    stem_index: dict[str, Path] = {}
    for p in pdb_dir.glob('*.pdb'):
        stem_index[p.stem.lower().replace(' ', '')] = p
    want = {str(x).strip().lower().replace(' ', '') for x in chain_ids if str(x).strip()}
    pdb_paths = sorted((stem_index[k] for k in want if k in stem_index), key=lambda x: x.stem)
    missing = len(want) - len(pdb_paths)
    if not pdb_paths:
        if verbose:
            print(f'No PDBs matched {len(want):,} requested ids under {pdb_dir}')
        return []
    if verbose:
        print(
            f'Loading {len(pdb_paths):,} PDBs for val list ({len(want):,} ids; '
            f'{missing:,} missing files or not in pdb_dir)'
        )
        print(f'MSA dir: {msa_dir}  ({len(list(msa_dir.glob("*.a3m"))):,} .a3m files)')

    samples, skipped = [], 0
    for pdb_path in tqdm(pdb_paths, desc='Loading PDBs', leave=False, disable=not verbose):
        cid = pdb_path.stem
        try:
            seq, coords, atom_mask = parse_backbone_pdb(pdb_path)
        except Exception:
            skipped += 1
            continue
        L = len(seq)
        if L < min_len or L > max_len:
            skipped += 1
            continue
        if not np.isfinite(coords[:, 0, :]).all():
            skipped += 1
            continue
        a3m_path = msa_dir / f'{cid}.a3m'
        cons, cov, has_msa = parse_a3m(a3m_path, L_target=L)
        samples.append(
            RNASample(
                seq=seq,
                coords=coords,
                chain_id=cid.upper(),
                atom_mask=atom_mask,
                msa_cons=cons,
                msa_cov=cov,
                has_msa=has_msa,
                atom_names=ATOM_NAMES,
            )
        )
    if verbose:
        n_msa = sum(1 for s in samples if s.has_msa)
        print(f'Loaded: {len(samples):,} chains  |  skipped: {skipped:,}')
        print(f'With real MSA: {n_msa:,} ({n_msa / max(len(samples), 1) * 100:.0f}%)')
    return samples


# ── Fold splits (list/ directory) ─────────────────────────────────────────────
def load_fold_splits(samples: list[RNASample], list_dir: Path,
                         val_fold: int = 0) -> tuple[list,list]:
    """
    Use published 10-fold cross-validation split files under list/.
    val_fold 0-9 selects which fold is validation; rest = train.
    Falls back to random 90/10 split if list files not found.
    """
    # Build lookup: lowercase chain_id → sample
    lookup: dict[str, RNASample] = {}
    for s in samples:
        cid = str(s.chain_id or '').lower().replace(' ','')
        lookup[cid] = s

    val_file = list_dir / f'valid_fold-{val_fold}'
    if not val_file.exists():
        print(f'[Split] Fold list not found at {val_file}, using random split')
        rng = random.Random(42); idx = list(range(len(samples))); rng.shuffle(idx)
        cut = max(1, int(len(idx)*0.1))
        val_idx = set(idx[:cut])
        return ([samples[i] for i in idx if i not in val_idx],
                [samples[i] for i in idx if i in val_idx])

    val_ids = set(val_file.read_text().strip().splitlines())
    val_ids = {v.strip().lower() for v in val_ids if v.strip()}

    # Train folds = all folds except val_fold
    train_ids: set[str] = set()
    for fold_i in range(10):
        if fold_i == val_fold: continue
        fold_file = list_dir / f'fold-{fold_i}_train_ids'
        if fold_file.exists():
            train_ids.update(l.strip().lower()
                             for l in fold_file.read_text().strip().splitlines()
                             if l.strip())

    train, val, neither = [], [], []
    for cid_lower, s in lookup.items():
        if   cid_lower in val_ids:   val.append(s)
        elif cid_lower in train_ids: train.append(s)
        else:                         neither.append(s)

    # Any chain not in any fold list goes to train
    train.extend(neither)
    print(f'[Split] fold-{val_fold} val  |  train={len(train):,}  val={len(val):,}')
    return train, val


# ─── §3  Dataset / collate ───────────────────────────────────────────────────
def encode_seq(seq: str) -> torch.Tensor:
    return torch.tensor([TOK.get(c,0) for c in str(seq).upper()], dtype=torch.long)

def one_hot(tok: torch.Tensor) -> torch.Tensor:
    return F.one_hot(tok.clamp(0,V-1), num_classes=V).float()


class RNADataset(Dataset):
    def __init__(self, samples, max_len=128, crop_mode='random', rng_seed=0):
        self.samples   = samples
        self.max_len   = max_len
        self.crop_mode = crop_mode
        self._rng      = random.Random(rng_seed)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ex   = self.samples[i]
        full = len(ex.seq)
        cap  = self.max_len
        if   full <= cap:                   start = 0
        elif self.crop_mode == 'random':    start = self._rng.randint(0, full-cap)
        else:                               start = max(0, (full-cap)//2)
        seq    = ex.seq[start:start+cap]
        L      = len(seq)
        tok    = encode_seq(seq)
        raw    = np.asarray(ex.coords, dtype=np.float64)
        if raw.ndim == 2: raw = raw[:,None,:]
        coords = torch.tensor(raw[start:start+L], dtype=torch.float32)
        am_np  = (np.asarray(ex.atom_mask, dtype=bool)[start:start+L]
                  if ex.atom_mask is not None
                  else np.isfinite(raw).all(axis=-1)[start:start+L])
        atom_mask = torch.tensor(am_np, dtype=torch.float32)
        # MSA features cropped to same window
        cons = (torch.tensor(ex.msa_cons[start:start+L], dtype=torch.float32)
                if ex.msa_cons is not None
                else torch.zeros(L, 4))
        cov  = (torch.tensor(ex.msa_cov[start:start+L, start:start+L], dtype=torch.float32)
                if ex.msa_cov is not None
                else torch.zeros(L, L))
        return dict(chain_id=ex.chain_id or f's{i}', seq=seq,
                    tok=tok, coords=coords, atom_mask=atom_mask,
                    cons=cons, cov=cov)


def collate_batch(batch):
    Ls  = [b['tok'].numel() for b in batch]; Lm = max(Ls); B = len(batch)
    Na  = batch[0]['coords'].shape[1]
    toks, coords_l, masks, atom_masks, cons_l, cov_l = [],[],[],[],[],[]
    for b in batch:
        L   = b['tok'].numel(); pad = Lm - L
        toks.append(F.pad(b['tok'], (0,pad)))
        coords_l.append(F.pad(b['coords'], (0,0,0,0,0,pad)))
        atom_masks.append(F.pad(b['atom_mask'], (0,0,0,pad)))
        m = torch.zeros(Lm); m[:L] = 1.0; masks.append(m)
        # MSA conservation (L,4) → pad to (Lm,4)
        cons_l.append(F.pad(b['cons'], (0,0,0,pad)))
        # MSA covariance (L,L) → pad to (Lm,Lm)
        cov_l.append(F.pad(b['cov'], (0,Lm-L, 0,Lm-L)))

    seq_t  = torch.stack(toks)       # (B,Lm)
    true_c = torch.stack(coords_l)   # (B,Lm,Na,3)
    atom_m = torch.stack(atom_masks) # (B,Lm,Na)
    mask   = torch.stack(masks)      # (B,Lm)
    cons_t = torch.stack(cons_l)     # (B,Lm,4)
    cov_t  = torch.stack(cov_l)      # (B,Lm,Lm)

    # ── Pair features (46 channels total) ──────────────────────────────────
    oh  = one_hot(seq_t)
    pi  = oh.unsqueeze(2).expand(B,Lm,Lm,V)   # (B,L,L,6)
    pj  = oh.unsqueeze(1).expand(B,Lm,Lm,V)   # (B,L,L,6)
    pos = torch.arange(Lm, dtype=torch.float32)
    rel = (pos.view(1,Lm,1) - pos.view(1,1,Lm)).expand(B,-1,-1)
    freq= torch.exp(torch.arange(8,dtype=torch.float32)*(-math.log(128)/7))
    ang = rel.unsqueeze(-1)*freq.view(1,1,1,8)
    re  = torch.cat([torch.sin(ang),torch.cos(ang)],dim=-1)    # (B,L,L,16)
    bb  = torch.tensor([0,1,2,3,4,6,8,12,16,24,32,48,64,96,128],dtype=torch.float32)
    bidx= torch.bucketize(rel.abs(),bb,right=False).long().clamp(0,15)
    rbkt= F.one_hot(bidx,num_classes=16).float()               # (B,L,L,16)
    sc_ch = cov_t.unsqueeze(-1)                                 # (B,L,L,1)  ← covariance
    # Total: 6+6+16+16+1 = 45 original + 1 (covariance replaces SS) = 46
    # (We keep SS signal through the covariance matrix which encodes base-pairs)
    pair_feat = torch.cat([pi,pj,re,rbkt,sc_ch],dim=-1)        # (B,L,L,46)

    return dict(seq_tokens=seq_t, pair_feat=pair_feat,
                true_coords=true_c, mask=mask, atom_mask=atom_m,
                msa_cons=cons_t)  # pass conservation for seq augmentation


def to_device(batch, device):
    return {k: v.to(device, non_blocking=(device.type=='cuda'))
            for k,v in batch.items() if isinstance(v,torch.Tensor)}


# ─── §4  Model ────────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len,d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]


class SeqAttnBlock(nn.Module):
    """Pre-norm MHSA + pair bias. fp32 softmax. -1e9 mask (finite gradient)."""
    def __init__(self,d_model,d_pair,n_heads,dropout=0.1):
        super().__init__()
        assert d_model%n_heads==0
        self.nh=n_heads; self.dh=d_model//n_heads; self.sc=self.dh**-0.5
        self.ln=nn.LayerNorm(d_model)
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.op=nn.Linear(d_model,d_model)
        self.pb=nn.Linear(d_pair,n_heads,bias=False)
        nn.init.zeros_(self.pb.weight)
        self.drop=nn.Dropout(dropout)
        self.fln=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(4*d_model,d_model))

    def forward(self,x,pair,mask):
        B,L,D=x.shape; h=self.ln(x)
        Q,K,Vv=self.qkv(h).chunk(3,dim=-1)
        Q=Q.view(B,L,self.nh,self.dh).transpose(1,2)
        K=K.view(B,L,self.nh,self.dh).transpose(1,2)
        Vv=Vv.view(B,L,self.nh,self.dh).transpose(1,2)
        with torch.autocast(device_type=x.device.type,enabled=False):
            attn=(Q.float()@K.float().transpose(-2,-1))*self.sc
            attn=attn+self.pb(pair.float()).permute(0,3,1,2)
            mask2d=(mask.unsqueeze(1)*mask.unsqueeze(2))
            attn=attn.masked_fill(mask2d.unsqueeze(1)==0,-1e9)
            attn=torch.nan_to_num(attn.clamp(-50,50),nan=0.)
            attn=self.drop(torch.nan_to_num(attn.softmax(dim=-1),nan=0.))
        out=(attn.to(Vv.dtype)@Vv).transpose(1,2).reshape(B,L,D)
        x=x+self.op(out)*mask.unsqueeze(-1)
        x=x+self.ff(self.fln(x))*mask.unsqueeze(-1)
        return x


class SeqTrunk(nn.Module):
    """8-layer pair-biased transformer.
    NEW: msa_proj adds conservation signal to seq embedding."""
    def __init__(self,d_model,d_pair,n_heads,n_layers,use_ckpt=False,max_len=512):
        super().__init__()
        self.use_ckpt=use_ckpt
        self.embed=nn.Embedding(V,d_model,padding_idx=0)
        self.pe=SinusoidalPE(d_model,max_len)
        # NEW: MSA conservation projection  (L,4) → (L,d_model)
        self.msa_proj=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,d_model))
        nn.init.zeros_(self.msa_proj[-1].weight)
        nn.init.zeros_(self.msa_proj[-1].bias)
        self.blocks=nn.ModuleList([SeqAttnBlock(d_model,d_pair,n_heads)
                                    for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d_model)

    def forward(self,tok,pair,mask,msa_cons=None):
        x=self.pe(self.embed(tok))
        # Add MSA conservation signal if available
        if msa_cons is not None:
            x=x+self.msa_proj(msa_cons.to(x.dtype))
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x=grad_ckpt.checkpoint(lambda x_,p_,m_=mask: blk(x_,p_,m_),
                                        x,pair,use_reentrant=False)
            else:
                x=blk(x,pair,mask)
        return self.ln(x)


class TriMult(nn.Module):
    """Triangle multiplicative update — fp32 einsum, zero-init output."""
    def __init__(self,d,mode='outgoing',dropout=0.05):
        super().__init__(); assert mode in ('outgoing','incoming'); self.mode=mode
        self.ln=nn.LayerNorm(d)
        self.L=nn.Linear(d,d); self.Lg=nn.Linear(d,d)
        self.R=nn.Linear(d,d); self.Rg=nn.Linear(d,d)
        self.out=nn.Linear(d,d); self.gt=nn.Linear(d,d)
        self.lo=nn.LayerNorm(d); self.drop=nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self,z,mask):
        m=(mask.unsqueeze(1)*mask.unsqueeze(2)).unsqueeze(-1)
        h=self.ln(z); a=self.L(h)*torch.sigmoid(self.Lg(h))
        b=self.R(h)*torch.sigmoid(self.Rg(h))
        af,bf=a.float(),b.float()
        if self.mode=='outgoing':
            prod=torch.einsum('bikd,bjkd->bijd',af,bf).to(z.dtype)
        else:
            prod=torch.einsum('bkid,bkjd->bijd',af,bf).to(z.dtype)
        n=max(mask.float().sum(-1).mean().item(),1.)
        prod=torch.nan_to_num(prod/n**0.5,nan=0.)
        g=torch.sigmoid(self.gt(h))
        return (z+self.drop(g*self.out(self.lo(prod))))*m


class PairTrans(nn.Module):
    def __init__(self,d,exp=4,dropout=0.05):
        super().__init__(); self.ln=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,exp*d),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(exp*d,d))
        nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)
    def forward(self,z,mask):
        m=(mask.unsqueeze(1)*mask.unsqueeze(2)).unsqueeze(-1)
        return (z+self.ff(self.ln(z)))*m


class PairUpdateBlock(nn.Module):
    def __init__(self,d_model,d_pair):
        super().__init__()
        self.op=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_pair//2),nn.GELU())
        self.oo=nn.Linear(d_pair//2,d_pair)
        nn.init.zeros_(self.oo.weight); nn.init.zeros_(self.oo.bias)
        self.to_=TriMult(d_pair,'outgoing'); self.ti=TriMult(d_pair,'incoming')
        self.tr=PairTrans(d_pair)

    def forward(self,seq,pair,mask):
        h=self.op(seq); outer=h.unsqueeze(2)+h.unsqueeze(1)
        m2=(mask.unsqueeze(1)*mask.unsqueeze(2)).unsqueeze(-1)
        pair=pair+self.oo(outer)*m2
        pair=self.to_(pair,mask); pair=self.ti(pair,mask); pair=self.tr(pair,mask)
        return pair


class PairEncoder(nn.Module):
    """Plain LayerNorm+MLP. in_dim=46 (45 original + 1 MSA covariance)."""
    def __init__(self,in_dim,d_pair):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(in_dim),
                                nn.Linear(in_dim,d_pair*2),nn.GELU(),
                                nn.Linear(d_pair*2,d_pair),nn.GELU(),
                                nn.Linear(d_pair,d_pair))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        B,L,_,C=x.shape
        return self.net(x.reshape(B*L*L,C)).reshape(B,L,L,-1)


def _ideal_helix(L,device):
    tw=math.radians(CFG['IDEAL_TWIST_DEG']); rise=CFG['IDEAL_RISE_ANG']; bond=CFG['IDEAL_C4_BOND']
    r=math.sqrt(max(bond**2-rise**2,0.01))/(2*math.sin(tw/2))
    pos=torch.arange(L,dtype=torch.float32,device=device)
    return torch.stack([r*torch.cos(tw*pos),r*torch.sin(tw*pos),rise*pos],dim=-1)

def _local_frame(c4):
    def _sn(v,dim=-1,eps=1e-6): return v/(v.pow(2).sum(dim=dim,keepdim=True).clamp(min=eps**2).sqrt())
    c4=c4.float()
    tang=torch.zeros_like(c4)
    tang[:,1:-1]=c4[:,2:]-c4[:,:-2]; tang[:,0]=c4[:,1]-c4[:,0]; tang[:,-1]=c4[:,-1]-c4[:,-2]
    tang=_sn(tang)
    ref_z=torch.zeros_like(tang); ref_z[...,2]=1.0
    ref_y=torch.zeros_like(tang); ref_y[...,1]=1.0
    dot_z=(tang*ref_z).sum(-1,keepdim=True).abs()
    alpha=((dot_z-0.9)/0.05).clamp(0.,1.)
    ref=ref_z*(1-alpha)+ref_y*alpha
    nv=_sn(torch.linalg.cross(tang,ref,dim=-1))
    bv=_sn(torch.linalg.cross(tang,nv,dim=-1))
    return tang,nv,bv


class TorsionCoordHead(nn.Module):
    """
    Absolute torsion IK → C4' coordinates.
    FIX: delta_scale = sigmoid(ds)*20 Å  (was softplus*80 → 68 Å → coord memorisation)
    FIX: tanh(raw)*pi  (was atan2(normalize) → NaN gradient at zero-init)
    FIX: absolute theta = helix_phase + offset  (was cumsum → error drift)
    """
    N_TORS=6
    def __init__(self,d_model,d_pair,n_atoms=9):
        super().__init__()
        self.n_atoms=n_atoms
        self.tors=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),
                                 nn.Dropout(0.05),nn.Linear(d_model,d_model//2),nn.GELU(),
                                 nn.Linear(d_model//2,self.N_TORS*2))
        nn.init.zeros_(self.tors[-1].weight); nn.init.zeros_(self.tors[-1].bias)
        self.c4r=nn.Sequential(nn.LayerNorm(d_model+d_pair),
                                nn.Linear(d_model+d_pair,d_model//2),nn.GELU(),
                                nn.Linear(d_model//2,3))
        nn.init.zeros_(self.c4r[-1].weight); nn.init.zeros_(self.c4r[-1].bias)
        self.ds=nn.Parameter(torch.tensor(-1.1))   # sigmoid(-1.1)*20 ≈ 5 Å at init
        if n_atoms>1:
            self.ah=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),
                                   nn.GELU(),nn.Linear(d_model,(n_atoms-1)*3))
            _io=torch.tensor([[-1.5,1.2,0.3],[1.2,0.8,0.2],[1.8,0.5,0.8],
                               [-1.2,0.9,-0.3],[-1.1,0.6,0.5],[0.3,1.4,-0.2],
                               [0.2,2.3,0.1],[0.1,3.6,0.4]])
            self.register_buffer('io',_io)
            nn.init.zeros_(self.ah[-1].weight); nn.init.zeros_(self.ah[-1].bias)

    def forward(self,seq,pair,mask):
        B,L,D=seq.shape; dev=seq.device
        raw=self.tors(seq).view(B,L,self.N_TORS,2)
        tw=math.radians(CFG['IDEAL_TWIST_DEG'])
        helix_phase=tw*torch.arange(L,dtype=torch.float32,device=dev)
        theta_offset=torch.tanh(raw[:,:,0,0])*math.pi
        theta=helix_phase.unsqueeze(0)+theta_offset
        bond=CFG['IDEAL_C4_BOND']; rise=CFG['IDEAL_RISE_ANG']
        r=math.sqrt(max(bond**2-rise**2,0.01))/(2*math.sin(tw/2))
        z_=rise*torch.arange(L,dtype=torch.float32,device=dev).unsqueeze(0).expand(B,-1)
        c4_ik=torch.stack([r*torch.cos(theta),r*torch.sin(theta),z_],dim=-1)
        pa=pair.mean(dim=2).to(seq.dtype)
        ni=torch.cat([seq,pa],dim=-1)
        # FIX: sigmoid*20 → bounded [0,20] Å  prevents coord memorisation
        delta_scale=torch.sigmoid(self.ds)*20.0
        delta=torch.tanh(self.c4r(ni).float())*delta_scale
        c4=(c4_ik.float()+delta)*mask.unsqueeze(-1).float()
        c4=c4.to(seq.dtype)
        if self.n_atoms==1: return c4.unsqueeze(2).to(seq.dtype)
        tang,nv,bv=_local_frame(c4)
        tang=tang.to(seq.dtype); nv=nv.to(seq.dtype); bv=bv.to(seq.dtype)
        ad=torch.tanh(self.ah(seq))*1.5; ad=ad.view(B,L,self.n_atoms-1,3)
        off=self.io.unsqueeze(0).unsqueeze(0).to(seq.dtype)+ad
        out=torch.zeros(B,L,self.n_atoms,3,device=dev,dtype=seq.dtype)
        out[:,:,0]=c4
        for a in range(self.n_atoms-1):
            out[:,:,a+1]=(c4+off[:,:,a,0:1]*tang+off[:,:,a,1:2]*nv+off[:,:,a,2:3]*bv)
        return out*mask.unsqueeze(-1).unsqueeze(-1)


class DistHead(nn.Module):
    def __init__(self,d_pair,n_bins=36):
        super().__init__(); self.n_bins=n_bins
        self.h=nn.Sequential(nn.LayerNorm(d_pair),nn.Linear(d_pair,d_pair),nn.GELU(),nn.Linear(d_pair,n_bins))
    def forward(self,pair):
        B,L,_,P=pair.shape; return self.h(pair.reshape(B*L*L,P)).reshape(B,L,L,self.n_bins)


class SSHead(nn.Module):
    def __init__(self,d_pair):
        super().__init__()
        self.h=nn.Sequential(nn.LayerNorm(d_pair),nn.Linear(d_pair,d_pair//2),nn.GELU(),nn.Linear(d_pair//2,1))
    def forward(self,pair): return self.h(pair).squeeze(-1)


class RNAFoldModel(nn.Module):
    def __init__(self,d_model=256,d_pair=128,n_seq_layers=8,n_pair_blocks=4,
                 n_heads=8,n_bins=36,use_grad_ckpt=False,n_atoms=9,max_len=512,
                 pair_in_ch=46):
        super().__init__()
        self.pair_in_ch = int(pair_in_ch)
        self.pair_enc  = PairEncoder(self.pair_in_ch, d_pair)
        self.seq_trunk = SeqTrunk(d_model,d_pair,n_heads,n_seq_layers,
                                   use_ckpt=use_grad_ckpt,max_len=max_len)
        self.pair_upd  = nn.ModuleList([PairUpdateBlock(d_model,d_pair)
                                         for _ in range(n_pair_blocks)])
        self.coord_h   = TorsionCoordHead(d_model,d_pair,n_atoms)
        self.dist_h    = DistHead(d_pair,n_bins)
        self.ss_h      = SSHead(d_pair)
        self.n_atoms   = n_atoms
        self.n_recycles = max(1,int(CFG.get('N_RECYCLES',3)))
        self.register_buffer('rec_edges',torch.linspace(2.,20.,steps=max(n_bins-1,1)))
        self.rec_proj  = nn.Sequential(nn.Linear(n_bins,d_pair,bias=False),nn.LayerNorm(d_pair))

    def _pass(self,seq,pair,mask):
        for pu in self.pair_upd: pair=pu(seq,pair,mask)
        return pair,self.coord_h(seq,pair,mask)

    def forward(self,seq_tok,pair_feat,mask,msa_cons=None):
        C = pair_feat.shape[-1]
        pic = self.pair_in_ch
        if C != pic:
            if C > pic:
                pair_feat = pair_feat[..., :pic]
            else:
                pad = pic - C
                pair_feat = F.pad(pair_feat, (0, pad))
        pair=self.pair_enc(pair_feat)
        seq=self.seq_trunk(seq_tok,pair,mask,msa_cons=msa_cons)
        pair,pred=self._pass(seq,pair,mask)
        for _ in range(self.n_recycles-1):
            c4=pred[:,:,0,:].float().detach()
            dmat=torch.cdist(c4,c4).clamp(2.,20.)
            bins=torch.bucketize(dmat,self.rec_edges.to(dmat.device),right=False)
            bins=bins.long().clamp(0,self.dist_h.n_bins-1)
            doh=F.one_hot(bins,num_classes=self.dist_h.n_bins).to(pair.dtype)
            pair=pair+self.rec_proj(doh)
            pair,pred=self._pass(seq,pair,mask)
        return dict(pred_coords=pred,dist_logits=self.dist_h(pair),
                    ss_logits=self.ss_h(pair),seq_repr=seq,pair_repr=pair)


def count_params(m): return f'{sum(p.numel() for p in m.parameters())/1e6:.2f}M'


# ─── §5  Losses ───────────────────────────────────────────────────────────────
_CS=6.1

def _sn(x,dim=-1,eps=1e-8): return x.pow(2).sum(dim=dim).clamp_min(eps).sqrt()
def _c4(c): return c[:,:,0,:] if c.dim()==4 else c

def kabsch_align(tx,px,mask):
    tx,px,mask=tx.float(),px.float(),mask.float(); B,L,_=tx.shape; out=px.clone()
    for b in range(B):
        m=mask[b]>0.5
        if m.sum()<3: continue
        X=torch.nan_to_num(tx[b,m]).float(); Y=torch.nan_to_num(px[b,m]).float()
        muX=X.mean(0,keepdim=True); Xc,Yc=X-muX,Y-Y.mean(0,keepdim=True)
        with torch.autocast(device_type=tx.device.type,enabled=False):
            C=Yc.t().float()@Xc.float()
            try: V2,_,Wt=torch.linalg.svd(C)
            except: continue
        R=V2.float()@Wt.float()
        if torch.det(R.float())<0: V2=V2.clone(); V2[:,-1]*=-1; R=V2.float()@Wt.float()
        out[b,m]=torch.nan_to_num((Yc.float()@R)+muX)
    return out

def kabsch_rmsd_loss(pred,true,mask):
    p=torch.nan_to_num(_c4(pred.float()),nan=0.,posinf=1e4,neginf=-1e4)/_CS
    t=torch.nan_to_num(_c4(true.float()),nan=0.,posinf=1e4,neginf=-1e4)/_CS
    pa=kabsch_align(t,p,mask.float()); m3=mask.float().unsqueeze(-1)
    mse=((pa-t).pow(2)*m3).sum()/(mask.sum()*3).clamp(1)
    rmsd=mse.sqrt()
    hub=(F.smooth_l1_loss(pa,t,beta=0.15,reduction='none')*m3).sum()/(mask.sum()*3).clamp(1)
    return rmsd+2.*hub, float(rmsd.detach()*_CS)

def bond_loss(pred,mask,tgt=6.1):
    c4=_c4(pred.float())/_CS; tn=tgt/_CS
    if c4.shape[1]<2: return c4.new_tensor(0.)
    d=_sn(c4[:,1:]-c4[:,:-1]); ms=mask[:,1:]*mask[:,:-1]
    return (F.smooth_l1_loss(d,torch.full_like(d,tn),beta=0.05,reduction='none')*ms).sum()/ms.sum().clamp(1)

def local_dist_loss(pred,true,mask,max_sep=12):
    pc=_c4(pred.float())/_CS; tc=_c4(true.float())/_CS; m=mask.float()
    if pc.shape[1]<2: return pc.new_tensor(0.)
    ls=[]
    for k in range(1,min(max_sep,pc.shape[1]-1)+1):
        pd=_sn(pc[:,k:]-pc[:,:-k]); td=_sn(tc[:,k:]-tc[:,:-k]); mk=m[:,k:]*m[:,:-k]
        if mk.sum()<1: continue
        ls.append((F.smooth_l1_loss(pd,td,beta=0.1,reduction='none')*mk).sum()/mk.sum().clamp(1))
    return torch.stack(ls).mean() if ls else pred.new_tensor(0.)

def backbone_shape_loss(pred,true,mask):
    def _ang(u,v): return torch.acos((F.normalize(u,dim=-1)*F.normalize(v,dim=-1)).sum(-1).clamp(-1,1))
    def _dih(a,b,c,d):
        a,b,c,d=a.float(),b.float(),c.float(),d.float()
        b0,b1,b2=a-b,c-b,d-c; b1n=F.normalize(b1,dim=-1)
        v=b0-(b0*b1n).sum(-1,keepdim=True)*b1n; w=b2-(b2*b1n).sum(-1,keepdim=True)*b1n
        return torch.atan2((torch.linalg.cross(b1n,v,dim=-1)*w).sum(-1),(v*w).sum(-1))
    pc=_c4(pred.float())/_CS; tc=_c4(true.float())/_CS; m=mask.float()
    if pc.shape[1]<4: return pc.new_tensor(0.),pc.new_tensor(0.)
    pa=_ang(pc[:,:-2]-pc[:,1:-1],pc[:,2:]-pc[:,1:-1])
    ta=_ang(tc[:,:-2]-tc[:,1:-1],tc[:,2:]-tc[:,1:-1])
    ma=m[:,:-2]*m[:,1:-1]*m[:,2:]
    al=((F.smooth_l1_loss(pa,ta,beta=0.05,reduction='none')*ma).sum()/ma.sum().clamp(1)
        if ma.sum()>0 else pc.new_tensor(0.))
    pt=_dih(pc[:,:-3],pc[:,1:-2],pc[:,2:-1],pc[:,3:])
    tt=_dih(tc[:,:-3],tc[:,1:-2],tc[:,2:-1],tc[:,3:])
    md=m[:,:-3]*m[:,1:-2]*m[:,2:-1]*m[:,3:]
    if md.sum()>0:
        sl=F.smooth_l1_loss(torch.sin(pt),torch.sin(tt),beta=0.08,reduction='none')
        cl=F.smooth_l1_loss(torch.cos(pt),torch.cos(tt),beta=0.08,reduction='none')
        dl=((sl+cl)*md).sum()/md.sum().clamp(1)
    else: dl=pc.new_tensor(0.)
    return al,dl

def pairwise_dist_loss(pred,true,mask):
    pc=torch.nan_to_num(_c4(pred.float()),nan=0.,posinf=1e4,neginf=-1e4)
    tc=torch.nan_to_num(_c4(true.float()),nan=0.,posinf=1e4,neginf=-1e4)
    B,L,_=pc.shape; pm=(mask>0.5).float()
    eye=torch.eye(L,device=pc.device).unsqueeze(0)
    pm2=pm.unsqueeze(2)*pm.unsqueeze(1)*(1-eye)
    dp=torch.cdist(pc,pc); dt=torch.cdist(tc,tc)
    sc=((dt*pm2).sum((1,2),keepdim=True)/pm2.sum((1,2),keepdim=True).clamp(1)).clamp_min(1.)
    err=F.smooth_l1_loss(dp/sc,dt/sc,beta=0.04,reduction='none')
    base=(err*pm2).sum()/pm2.sum().clamp(1)
    sep=(torch.arange(L,device=pc.device).view(1,L,1)-
         torch.arange(L,device=pc.device).view(1,1,L)).abs().float()
    lr_mask=pm2*(sep>24).float()
    if lr_mask.sum()>0:
        return base+(err*lr_mask).sum()/lr_mask.sum().clamp(1)
    return base

def atom_dist_loss(pred,true,mask):
    p,t,m=pred.float(),true.float(),mask.float()
    if p.dim()!=4: return p.new_tensor(0.)
    B,L,Na,_=p.shape
    if Na<2: return p.new_tensor(0.)
    pi=torch.cdist(p.reshape(B*L,Na,3),p.reshape(B*L,Na,3)).reshape(B,L,Na,Na)
    ti=torch.cdist(t.reshape(B*L,Na,3),t.reshape(B*L,Na,3)).reshape(B,L,Na,Na)
    wm=m.unsqueeze(-1).unsqueeze(-1).expand_as(pi)
    li=(F.smooth_l1_loss(pi,ti,beta=0.25,reduction='none')*wm).sum()/wm.sum().clamp(1)
    if L<2: return li
    pa=_sn(p[:,1:]-p[:,:-1]); ta=_sn(t[:,1:]-t[:,:-1])
    wa=(m[:,1:]*m[:,:-1]).unsqueeze(-1).expand_as(pa)
    la=(F.smooth_l1_loss(pa,ta,beta=0.25,reduction='none')*wa).sum()/wa.sum().clamp(1)
    return 0.5*(li+la)

def dist_nll_loss(dl,true,mask,n_bins,dlo=2.,dhi=20.):
    tc=torch.nan_to_num(_c4(true.float()),nan=0.); B,L,_=tc.shape
    fm=mask*torch.isfinite(tc).all(-1).float()
    pm=fm.unsqueeze(1)*fm.unsqueeze(2)*(1-torch.eye(L,device=tc.device).unsqueeze(0))
    d=torch.cdist(tc,tc)
    idx=((d-dlo)/max(dhi-dlo,1e-8)*n_bins).floor().long().clamp(0,n_bins-1)
    nll=-dl.log_softmax(-1).gather(-1,idx.unsqueeze(-1)).squeeze(-1)
    return (nll*pm).sum()/pm.sum().clamp(1)

def rg_loss(pred,mask):
    c4=_c4(pred.float()); m=mask.float().unsqueeze(-1)
    if mask.sum()<2: return c4.new_tensor(0.)
    ctr=(c4*m).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp(1)
    rg=((c4-ctr).pow(2)*m).sum()/(mask.sum()*3).clamp(1)
    return rg/_CS**2

def clash_loss(pred,mask):
    c4=_c4(pred.float()); B,L,_=c4.shape
    if L<3: return c4.new_tensor(0.)
    pm=(mask>0.5).float()
    pm2=pm.unsqueeze(2)*pm.unsqueeze(1)
    eye=torch.eye(L,device=c4.device).unsqueeze(0)
    sep=(torch.arange(L,device=c4.device).view(1,L,1)-
         torch.arange(L,device=c4.device).view(1,1,L)).abs().float()
    non_bond=pm2*(1-eye)*(sep>1).float()
    d=torch.cdist(c4,c4)
    return (F.relu(3.5-d)*non_bond).sum()/non_bond.sum().clamp(1)/_CS


def combined_loss(out,batch,epoch,n_bins):
    pred=torch.nan_to_num(out['pred_coords'].float(),nan=0.,posinf=1e4,neginf=-1e4)
    true=torch.nan_to_num(batch['true_coords'].float(),nan=0.,posinf=1e4,neginf=-1e4)
    msk=batch['mask'].float(); logs={}
    def _c(x): return torch.nan_to_num(x,nan=0.,posinf=50.).clamp(max=50.)
    gr=min(1.0,max(0.25,epoch/max(CFG['GEOM_FULL_EPOCH'],1)))

    cl,rmsd_A=kabsch_rmsd_loss(pred,true,msk); cl=_c(cl)
    logs['c4_rmsd_A']=rmsd_A; total=CFG['W_RMSD']*cl

    bl=_c(bond_loss(pred,msk)); ld=_c(local_dist_loss(pred,true,msk))
    al,dhl=backbone_shape_loss(pred,true,msk); al,dhl=_c(al),_c(dhl)
    at=_c(atom_dist_loss(pred,true,msk))
    total=total+gr*(CFG['W_BOND']*bl+CFG['W_LOCAL_DIST']*ld+
                    CFG['W_BB_ANGLE']*al+CFG['W_BB_DIHEDRAL']*dhl+CFG['W_ATOM_DIST']*at)
    logs.update({'bond':float(bl.detach()),'ld':float(ld.detach()),
                 'ang':float(al.detach()),'at':float(at.detach())})

    dnll=_c(dist_nll_loss(out['dist_logits'],true,msk,n_bins))
    total=total+CFG['W_DIST_NLL']*dnll

    if epoch>=CFG['PHASE2_START']:
        r=min(1.,(epoch-CFG['PHASE2_START']+1)/3.)
        pd=_c(pairwise_dist_loss(pred,true,msk))
        total=total+r*CFG['W_PAIR_DIST']*pd; logs['pd']=float(pd.detach())

    ssl=_c(F.binary_cross_entropy_with_logits(
        out['ss_logits'],
        (torch.cdist(_c4(true),_c4(true))<9.).float()*
        (msk.unsqueeze(1)*msk.unsqueeze(2))))
    total=total+min(1.,epoch/2.)*CFG['W_SS_AUX']*ssl; logs['ss']=float(ssl.detach())

    rgl=_c(rg_loss(pred,msk))
    total=total+CFG['W_RG']*rgl

    cll=_c(clash_loss(pred,msk))
    total=total+CFG['W_CLASH']*cll

    total=torch.nan_to_num(total,nan=1e4,posinf=1e4).clamp(max=1e4)
    logs['loss']=float(total.detach())
    return total,logs


# ─── §6  Evaluation ───────────────────────────────────────────────────────────
def _kabsch_np(X,Y):
    X,Y=np.asarray(X,np.float64),np.asarray(Y,np.float64)
    mX,mY=X.mean(0),Y.mean(0); Xc,Yc=X-mX,Y-mY
    V2,_,Wt=np.linalg.svd(Yc.T@Xc); R=V2@Wt
    if np.linalg.det(R)<0: V2=V2.copy(); V2[:,-1]*=-1; R=V2@Wt
    return R.astype(np.float32),mX.astype(np.float32),mY.astype(np.float32)

def evaluate_sample(model,sample,device=None,save_pdb=False,out_dir=Path('.')):
    """RMSD matches training/val ``c4_rmsd_A`` from ``kabsch_rmsd_loss`` (not √(mean ||Δ||²) per residue)."""
    device=device or DEVICE; model.eval()
    with torch.no_grad():
        ds=RNADataset([sample],max_len=CFG['MAX_LEN'],crop_mode='center')
        batch=collate_batch([ds[0]]); batch=to_device(batch,device)
        msa_cons=batch.get('msa_cons')
        out=model(batch['seq_tokens'],batch['pair_feat'],batch['mask'],msa_cons=msa_cons)
        pred=torch.nan_to_num(out['pred_coords'].float(),nan=0.,posinf=1e4,neginf=-1e4)
        true=torch.nan_to_num(batch['true_coords'].float(),nan=0.,posinf=1e4,neginf=-1e4)
        msk=batch['mask'].float()
        _,rmsd_A=kabsch_rmsd_loss(pred,true,msk)
    L=int(batch['mask'][0].sum().item())
    tc=_c4(true)[0,:L].cpu().numpy()
    pc=_c4(pred)[0,:L].cpu().numpy()
    R,mX,mY=_kabsch_np(tc,pc); pa=(pc-mY)@R+mX
    pr=np.sqrt(((tc-pa)**2).sum(-1))
    d0=max(0.5,1.24*(max(L,16)-15)**(1/3)-1.8)
    tm=float(np.mean(1/(1+(pr/d0)**2)))
    return dict(rmsd=float(rmsd_A),tm=tm,L=L,chain_id=sample.chain_id)

def batch_evaluate(model,samples,device,max_chains=200,silent=False):
    model.eval(); rows=[]
    show = not silent
    for s in tqdm(samples[:max_chains], desc='val eval', leave=False, disable=not show):
        try:
            m=evaluate_sample(model,s,device,save_pdb=False)
            rows.append(m)
            if show:
                print(f'  L={m["L"]:4d} | RMSD={m["rmsd"]:.3f} A | TM={m["tm"]:.4f}')
        except Exception as e:
            if show:
                print(f'  [WARN] {s.chain_id}: {e}')
    import pandas as pd
    df=pd.DataFrame(rows)
    if not df.empty and show:
        print(f'\n=== Val ({len(df)} chains) ===')
        print(f'Mean RMSD   : {df.rmsd.mean():.3f} A')
        print(f'Median RMSD : {df.rmsd.median():.3f} A')
        print(f'<= 3 A      : {(df.rmsd<=3.0).mean()*100:.1f}%')
        print(f'<= 5 A      : {(df.rmsd<=5.0).mean()*100:.1f}%')
        print(f'Mean TM     : {df.tm.mean():.4f}')
        print(f'Median TM   : {df.tm.median():.4f}')
        for lo,hi,label in [(0,32,'short 1-32'),(33,64,'mid 33-64'),
                             (65,96,'mid 65-96'),(97,200,'long 97+')]:
            sub=df[(df.L>=lo)&(df.L<=hi)]
            if not sub.empty:
                print(f'  {label:12s} n={len(sub):3d} | '
                      f'mean={sub.rmsd.mean():.2f}A  TM={sub.tm.mean():.3f}  '
                      f'<=3A={( sub.rmsd<=3).mean()*100:.0f}%')
    return df


# ─── §7  Checkpoint helpers ───────────────────────────────────────────────────
def _save(path,model,opt,sched,scaler,ep,meta=None):
    torch.save({'model':model.state_dict(),'opt':opt.state_dict(),
                'sched':sched.state_dict(),'scaler':scaler.state_dict(),
                'epoch':ep,'meta':meta or {}},path)

def _load(path,model,opt,sched,scaler,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(ck['model'],strict=False)  # strict=False: ok if MSA proj is new
    for o,k in [(opt,'opt'),(sched,'sched'),(scaler,'scaler')]:
        try: o.load_state_dict(ck[k])
        except: pass
    return ck.get('epoch',0)


# ─── §8  Training loop ────────────────────────────────────────────────────────
def _len_cap(ep):
    if ep<=CFG['CURR_P1_EPOCHS']: return 32
    if ep<=CFG['CURR_P2_EPOCHS']: return 64
    if ep<=CFG['CURR_P3_EPOCHS']: return 96
    return 128

def run_training(model,train_s,val_s,device):
    import pandas as pd
    epochs=CFG['FULL_EPOCHS']; lr=CFG['LR']; batch_size=CFG['BATCH_SIZE']
    ck_dir=CKPT_DIR; log_path=ck_dir/'train_log.jsonl'
    csv_path=ck_dir/'metrics.csv'

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.type=='cuda': torch.cuda.manual_seed_all(42)

    amp_on=CFG['USE_AMP'] and device.type=='cuda'
    n_bins=model.dist_h.n_bins

    opt=torch.optim.AdamW(model.parameters(),lr=lr,
                           weight_decay=CFG['WEIGHT_DECAY'],betas=(0.9,0.999),eps=1e-8)
    scaler=(GradScaler(device.type,enabled=amp_on) if _AMP_NEW else GradScaler(enabled=amp_on))

    _wu=CFG['LR_WARMUP_STEPS']; _eta=CFG['LR_ETA_MIN']
    _tot=max(epochs*(max(len(train_s)//max(batch_size,1),1)),1)
    def _lrl(step):
        if step<_wu: return max(0.01,step/max(_wu,1))
        p=(step-_wu)/max(_tot-_wu,1)
        return max(_eta/lr,0.5*(1+math.cos(math.pi*min(p,1.))))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,_lrl)
        sched.step()
    step=0

    # Resume if checkpoint exists (prefer last.pt; legacy last_model.pt still works)
    _resume = os.environ.get('RNA_RESUME')
    if _resume:
        rp = Path(_resume)
    else:
        rp = ck_dir / 'last.pt'
        if not rp.is_file():
            rp = ck_dir / 'last_model.pt'
    start_ep=0
    if rp.is_file():
        start_ep=_load(rp,model,opt,sched,scaler,device)
        print(f'Resumed from {rp} (epoch {start_ep})')

    best_rmsd=float('inf'); best_path=ck_dir/'best.pt'; last_path=ck_dir/'last.pt'
    named_p=list(model.named_parameters())

    def _sani():
        nb=0
        for _,p in named_p:
            if p.grad is None: continue
            bm=~torch.isfinite(p.grad)
            if bm.any():
                nb+=int(bm.sum()); p.grad=torch.nan_to_num(p.grad,nan=0.,posinf=0.,neginf=0.)
        return nb

    print(f'Train {len(train_s):,} | Val {len(val_s):,} | params={count_params(model)} | '
          f'epochs={epochs} | lr={lr:.2e} | batch={batch_size} | amp={amp_on} | '
          f'recycles={model.n_recycles}')

    for ep in range(start_ep+1,epochs+1):
        cap=_len_cap(ep)
        ep_tr=[s for s in train_s if len(s.seq)<=cap] or train_s
        ds_tr=RNADataset(ep_tr,max_len=cap,crop_mode='random',rng_seed=1337+ep)
        dl_tr=DataLoader(ds_tr,batch_size=batch_size,shuffle=True,
                          collate_fn=collate_batch,pin_memory=(device.type=='cuda'),
                          num_workers=CFG['NUM_WORKERS'])
        ep_va=[s for s in val_s if len(s.seq)<=cap] or val_s
        ds_va=RNADataset(ep_va,max_len=cap,crop_mode='center',rng_seed=0)
        dl_va=DataLoader(ds_va,batch_size=batch_size,shuffle=False,
                          collate_fn=collate_batch,pin_memory=(device.type=='cuda'),
                          num_workers=CFG['NUM_WORKERS'])

        model.train(); run=defaultdict(float); n_s=skipped=nan_g_total=0
        pbar=tqdm(dl_tr,desc=f'ep{ep}/{epochs} L<={cap}',leave=False)
        for batch in pbar:
            try:
                batch=to_device(batch,device)
                msa_cons=batch.get('msa_cons')
                opt.zero_grad(set_to_none=True)
                amp_ctx=(autocast(device.type,dtype=_ADTYPE,enabled=amp_on) if _AMP_NEW
                         else autocast(enabled=amp_on))
                with amp_ctx:
                    out=model(batch['seq_tokens'],batch['pair_feat'],
                              batch['mask'],msa_cons=msa_cons)
                    loss,logs=combined_loss(out,batch,ep,n_bins)
                if not torch.isfinite(loss): skipped+=1; continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt); nb=_sani(); nan_g_total+=nb
                torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['GRAD_CLIP'])
                sb=scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale()>=sb: step+=1; sched.step()
                n_s+=1
                for k,v in logs.items(): run[k]+=float(v)
                if n_s%20==0:
                    d=max(n_s,1)
                    pbar.set_postfix(loss=f'{run["loss"]/d:.3f}',
                                     rmsd=f'{run.get("c4_rmsd_A",0)/d:.2f}A',
                                     nan_g=nan_g_total,skip=skipped)
            except (torch.cuda.OutOfMemoryError,RuntimeError) as e:
                if 'out of memory' not in str(e).lower(): raise
                skipped+=1; opt.zero_grad(set_to_none=True)
                if device.type=='cuda': torch.cuda.empty_cache()

        d=max(n_s,1); lc=sched.get_last_lr()[0] if hasattr(sched,'get_last_lr') else lr
        print(f'ep{ep:3d}/{epochs} L<={cap} | loss={run["loss"]/d:.4f} '
              f'rmsd={run.get("c4_rmsd_A",0)/d:.3f}A '
              f'bond={run.get("bond",0)/d:.4f} ld={run.get("ld",0)/d:.4f} '
              f'skip={skipped} nan_g={nan_g_total} lr={lc:.2e}')

        # Val
        model.eval(); vrun=defaultdict(float); vn=0
        with torch.no_grad():
            for batch in dl_va:
                try:
                    batch=to_device(batch,device); msa_cons=batch.get('msa_cons')
                    amp_ctx=(autocast(device.type,dtype=_ADTYPE,enabled=amp_on) if _AMP_NEW
                             else autocast(enabled=amp_on))
                    with amp_ctx:
                        out=model(batch['seq_tokens'],batch['pair_feat'],
                                  batch['mask'],msa_cons=msa_cons)
                        loss,logs=combined_loss(out,batch,ep,n_bins)
                    if torch.isfinite(loss):
                        for k,v in logs.items(): vrun[k]+=float(v); vn+=1
                except (torch.cuda.OutOfMemoryError,RuntimeError) as e:
                    if 'out of memory' not in str(e).lower(): raise
                    if device.type=='cuda': torch.cuda.empty_cache()
        if vn:
            vd=max(vn,1); vrmsd=vrun.get('c4_rmsd_A',0)/vd
            print(f'  VAL L<={cap} | loss={vrun["loss"]/vd:.4f} rmsd={vrmsd:.3f}A '
                  f'bond={vrun.get("bond",0)/vd:.4f} ld={vrun.get("ld",0)/vd:.4f}')
            if vrmsd<best_rmsd:
                best_rmsd=vrmsd
                _save(best_path,model,opt,sched,scaler,ep,{'val_rmsd_A':vrmsd,'len_cap':cap})
                print(f'  -> BEST (val RMSD={vrmsd:.3f} A at L<={cap})')
            with log_path.open('a') as lf:
                lf.write(json.dumps({'ep':ep,'cap':cap,
                                      'tr_rmsd':run.get('c4_rmsd_A',0)/d,
                                      'va_rmsd':vrmsd})+'\n')
        _save(last_path,model,opt,sched,scaler,ep,{'best_rmsd':best_rmsd})

    print(f'Done. best_val_rmsd={best_rmsd:.3f} A')


# ─── §9  Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n'+'='*60)
    print('Loading structure dataset ...')
    all_samples = load_structure_samples(PDB_DIR, MSA_DIR, verbose=True)
    print(f'Total loaded: {len(all_samples):,} chains')

    print('\nBuilding train/val splits (fold-0 = val) ...')
    train_s, val_s = load_fold_splits(all_samples, LIST_DIR, val_fold=0)
    print(f'Train: {len(train_s):,}  |  Val: {len(val_s):,}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    print('\nBuilding model ...')
    model = RNAFoldModel(
        d_model=CFG['D_MODEL'], d_pair=CFG['D_PAIR'],
        n_seq_layers=CFG['N_SEQ_LAYERS'], n_pair_blocks=CFG['N_PAIR_BLOCKS'],
        n_heads=CFG['N_HEADS'], n_bins=CFG['N_BINS'],
        use_grad_ckpt=CFG['USE_GRAD_CKPT'], n_atoms=N_ATOMS,
        max_len=CFG['MAX_LEN']+16,
        pair_in_ch=46,
    ).to(DEVICE)
    print(f'Model params: {count_params(model)}')
    if torch.cuda.is_available():
        print(f'GPU mem after init: {torch.cuda.memory_allocated()/1e6:.1f} MB')

    # Sanity check
    if val_s:
        short=[s for s in val_s if len(s.seq)<=32]
        if short:
            print('\nSanity check (before training):')
            m0=evaluate_sample(model,short[0],DEVICE,save_pdb=False)
            print(f'  L={m0["L"]:3d} | RMSD={m0["rmsd"]:.2f} A (helix prior, expect 6-15 A)')

    print('\nStarting training ...')
    run_training(model, train_s, val_s, DEVICE)

    # Load best checkpoint
    best_path = CKPT_DIR / 'best.pt'
    if not best_path.is_file():
        best_path = CKPT_DIR / 'best_model.pt'
    if best_path.is_file():
        ck = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        rv = ck.get('meta',{}).get('val_rmsd_A')
        print(f'Loaded {best_path.name}' + (f' (val RMSD={rv:.3f} A)' if isinstance(rv,float) else ''))

    print('\nFull validation evaluation ...')
    df = batch_evaluate(model, val_s, DEVICE, max_chains=200, silent=True)
    df.to_csv(EVAL_OUT / 'val_results.csv', index=False)