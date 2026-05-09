#!/usr/bin/env bash
# amarel_train_singularity.sh  v12-msa
# ─────────────────────────────────────────────
# USAGE:
#   module load singularity
#   export RNA3D_HOST=/scratch/$USER/rna3d
#   export SIF_PATH=/scratch/$USER/containers/pytorch-cuda12.sif
#   bash scripts/amarel_train_singularity.sh
#
# RESUME:
#   export RNA_RESUME=/rna3d/checkpoints_v12/last_model.pt
#   bash scripts/amarel_train_singularity.sh

set -euo pipefail

# ── 1. Find singularity ────────────────────────────────────────────────────────
SINGULARITY_CMD=""
command -v singularity &>/dev/null && SINGULARITY_CMD="singularity"
command -v apptainer  &>/dev/null && SINGULARITY_CMD="${SINGULARITY_CMD:-apptainer}"
[[ -z "${SINGULARITY_CMD}" ]] && { echo "ERROR: run: module load singularity"; exit 1; }

# ── 2. Find repo ───────────────────────────────────────────────────────────────
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RNA3D_HOST="${RNA3D_HOST:-$HOME/rna3d}"
[[ -f "${RNA3D_HOST}/rna.py" ]] || { [[ -f "${_REPO}/rna.py" ]] && RNA3D_HOST="${_REPO}"; }
[[ -f "${RNA3D_HOST}/rna.py" ]] || { echo "ERROR: rna.py not found in ${RNA3D_HOST}"; exit 1; }
echo "INFO: host=${RNA3D_HOST}"

RNA3D_INNER="/rna3d"
SIF_PATH="${SIF_PATH:?Set SIF_PATH}"
BIND_ARGS=("--nv" "-B" "${RNA3D_HOST}:${RNA3D_INNER}")

# Bind Infernal if available
INFERNAL_HOST="${INFERNAL_HOST:-$HOME/software/infernal}"
[[ -d "${INFERNAL_HOST}/bin" ]] && BIND_ARGS+=("-B" "${INFERNAL_HOST}:/infernal") && echo "INFO: Infernal bound"

# ── 3. Force correct settings (override stale env vars) ───────────────────────
RNA_USE_AMP="${RNA_USE_AMP:-1}"
RNA_USE_GRAD_CKPT="${RNA_USE_GRAD_CKPT:-0}"
RNA_BATCH_SIZE="${RNA_BATCH_SIZE:-4}"
RNA_N_RECYCLES="${RNA_N_RECYCLES:-3}"
RNA_LR="${RNA_LR:-3e-4}"
RNA_GRAD_CLIP="${RNA_GRAD_CLIP:-0.5}"
RNA_GRAD_ACCUM="${RNA_GRAD_ACCUM:-4}"
RNA_FULL_EPOCHS="${RNA_FULL_EPOCHS:-100}"

# Auto-fix known-bad stale values
[[ "${RNA_USE_AMP}" == "0" ]]       && echo "WARN: RNA_USE_AMP=0 → forcing 1"      && RNA_USE_AMP="1"
[[ "${RNA_BATCH_SIZE}" == "2" ]]    && echo "WARN: RNA_BATCH_SIZE=2 → forcing 4"   && RNA_BATCH_SIZE="4"
[[ "${RNA_N_RECYCLES}" == "1" ]]    && echo "WARN: RNA_N_RECYCLES=1 → forcing 3"   && RNA_N_RECYCLES="3"
[[ "${RNA_LR}" == "1e-4" ]]         && echo "WARN: RNA_LR=1e-4 → forcing 3e-4"     && RNA_LR="3e-4"

echo "INFO: AMP=${RNA_USE_AMP}  BATCH=${RNA_BATCH_SIZE}  RECYCLES=${RNA_N_RECYCLES}  LR=${RNA_LR}  EPOCHS=${RNA_FULL_EPOCHS}"
export RNA3D_ROOT="${RNA3D_INNER}"

# ── 4. Build env block ─────────────────────────────────────────────────────────
ENV_BLOCK="export RNA3D_ROOT='${RNA3D_INNER}'; export PYTHONUNBUFFERED=1"
for _var in \
    RNA_LR RNA_BATCH_SIZE RNA_N_RECYCLES RNA_GRAD_CLIP RNA_GRAD_ACCUM RNA_FULL_EPOCHS \
    RNA_USE_AMP RNA_USE_GRAD_CKPT \
    RNA_W_RMSD RNA_W_BOND RNA_W_LOCAL_DIST RNA_W_BB_ANGLE RNA_W_BB_DIHEDRAL \
    RNA_W_PAIR_DIST RNA_W_SS_AUX RNA_W_ATOM_DIST RNA_W_DIST_NLL RNA_W_RG RNA_W_CLASH \
    RNA_DATASET_DIR RNA_CHECKPOINT_DIR RNA_EVAL_OUT RNA_RESUME; do
  _val="${!_var:-}"
  [[ -n "${_val}" ]] && ENV_BLOCK+="; export ${_var}='${_val}'"
done

# ── 5. Launch ─────────────────────────────────────────────────────────────────
"${SINGULARITY_CMD}" exec "${BIND_ARGS[@]}" "${SIF_PATH}" bash -lc "
  set -e
  ${ENV_BLOCK}
  cd '${RNA3D_INNER}'
  python3 rna.py
"
echo "Done."