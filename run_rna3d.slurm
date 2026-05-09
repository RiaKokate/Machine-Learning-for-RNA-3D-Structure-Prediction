#!/bin/bash
#SBATCH --job-name=rna3d-dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/rk1285/rna3d/logs/%x-%j.out
#SBATCH --error=/scratch/rk1285/rna3d/logs/%x-%j.err

module purge
module load singularity cuda cudnn

cd /scratch/rk1285/rna3d
mkdir -p logs

export RNA3D_HOST=/scratch/rk1285/rna3d
export SIF_PATH="$HOME/containers/pytorch.sif"
export RNA_CHECKPOINT_DIR=/rna3d/checkpoints_v11_dev_id50
export RNA_EVAL_OUT=/rna3d/eval_out_v11_dev_id50
export RNA_SPLIT_MAX_SEQ_ID=0.50
export RNA_FAMILY_SEQ_ID=0.70

bash scripts/amarel_train_singularity.sh
