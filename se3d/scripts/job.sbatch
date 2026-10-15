#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/vinayakg/se3d/logs/slurm/%x-%j.out
# Generic se3d GPU job. Submit through se3d/scripts/submit.sh, which picks the
# partition / GPU type. The command to run is passed in $SE3D_CMD.
set -eo pipefail
R=/fs/vulcan-scratch/vinayakg/cmsc673-sphere-encoder
S=/fs/nexus-scratch/vinayakg/se3d
export PYTHONPATH=$R
export HF_HOME=$S/cache/hf
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export TORCH_HOME=$S/cache/torch
export TORCH_EXTENSIONS_DIR=$S/cache/torch_ext
export PIP_CACHE_DIR=$S/cache/pip
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export CUDA_HOME=/opt/common/cuda/cuda-12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export PATH=$S/envs/se3d/bin:$PATH
cd $R
echo "+ host $(hostname)  job $SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "+ cmd: $SE3D_CMD"
bash -c "$SE3D_CMD"
echo "+ JOB_DONE $(date)"
