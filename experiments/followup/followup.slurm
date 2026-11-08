#!/usr/bin/env bash
#SBATCH --job-name=intrinsics-followup
#SBATCH --account=cbcb
#SBATCH --partition=cbcb
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Usage: sbatch followup.slurm ROOT PAIRS CHECKPOINT TASK VARIANT METHOD SEED OUTPUT [extra args]
set -euo pipefail
project_root=$1
pairs_path=$2
checkpoint_path=$3
task_name=$4
variant_name=$5
movement_method=$6
run_seed=$7
result_dir=$8
shift 8
cd "$project_root"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi
sha256sum experiments/followup/adaptive_transform.py experiments/followup/compare_adaptive.py
"$project_root/.venv-gpu/bin/python" -m experiments.followup.compare_adaptive \
    --pairs "$pairs_path" --checkpoint "$checkpoint_path" --task "$task_name" \
    --variants "$variant_name" --method "$movement_method" --seeds "$run_seed" \
    --output-dir "$result_dir" --steps 400 --batch-size 1 --accumulation 4 "$@"
