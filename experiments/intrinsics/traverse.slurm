#!/usr/bin/env bash
#SBATCH --job-name=intrinsics-traversal
#SBATCH --account=cbcb
#SBATCH --partition=cbcb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
if [[ $# -lt 4 ]]; then
    echo 'Usage: sbatch traverse.slurm PROJECT_ROOT CHECKPOINT OUTPUT_DIR IMAGE [IMAGE ...]'
    exit 2
fi
PROJECT_ROOT=$1
CHECKPOINT_PATH=$2
RESULT_DIR=$3
shift 3
cd "$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi
"$PROJECT_ROOT/.venv-gpu/bin/python" -m experiments.intrinsics.traverse \
    --checkpoint "$CHECKPOINT_PATH" --image "$@" --output-dir "$RESULT_DIR" \
    --encoder-from-checkpoint --seed 673 --dtype float32
