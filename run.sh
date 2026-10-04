#!/usr/bin/env bash
#
# usage:
# ./run.sh [--dist_mode local|distributed] [--cache_dir workspace/cache] script.py --arg1 val1

# parse arguments
DIST_MODE="local"  # local | distributed
CACHE_DIR="./workspace/cache"  # huggingface / torch hub cache
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dist-mode|--dist_mode)
      DIST_MODE="$2"
      shift 2
      ;;
    --dist-mode=*|--dist_mode=*)
      DIST_MODE="${1#*=}"
      shift 1
      ;;
    --cache-dir|--cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --cache-dir=*|--cache_dir=*)
      CACHE_DIR="${1#*=}"
      shift 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# restore positional arguments (script and its args)
set -- "${POSITIONAL_ARGS[@]}"

# inputs
INPUT_SCRIPT=$1
INPUT_ARGVS=${@:2}

echo "+ DIST_MODE: $DIST_MODE"
echo "+ CACHE_DIR: $CACHE_DIR"
echo "+ INPUT_SCRIPT: $INPUT_SCRIPT"
echo "+ INPUT_ARGVS: $INPUT_ARGVS"

# if the input script is not found, exit
if [ ! -f "$INPUT_SCRIPT" ]; then
    echo "$INPUT_SCRIPT not found"
    exit 1
fi

nvidia-smi

# envs
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$PWD

export HF_HOME=$CACHE_DIR/huggingface
export TORCH_HOME=$CACHE_DIR/torch 
echo "+ HF_HOME: $HF_HOME"
echo "+ TORCH_HOME: $TORCH_HOME"

# nGPUs
NGPUS=$(nvidia-smi -L | wc -l)
echo "+ NGPUS: $NGPUS"
if [ $NGPUS -eq 0 ]; then
    echo "No GPU found"
    exit 1
fi

# cmd
if [ "$DIST_MODE" == "distributed" ]; then
    # slurm distributed setup
    NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    HEAD_NODE=${NODES_ARRAY[0]}
    HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')
    WORLD_SIZE=$SLURM_JOB_NUM_NODES
    SRUN_CMD="srun"

    echo "+ NODES_ARRAY: ${NODES_ARRAY[@]}"
    echo "+ HEAD_NODE: $HEAD_NODE"
else
    # local setup
    HEAD_NODE_IP="127.0.0.1"
    WORLD_SIZE=1
    SRUN_CMD=""

    echo "+ Running in LOCAL mode"
fi
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close();')

echo "+ HEAD_NODE_IP: $HEAD_NODE_IP"
echo "+ WORLD_SIZE: $WORLD_SIZE"

TORCHRUN_CMD="torchrun --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $HEAD_NODE_IP:$PORT --nnode $WORLD_SIZE --nproc_per_node $NGPUS"
PREFIX_CMD="$SRUN_CMD $TORCHRUN_CMD"


# fire up
set -x
$PREFIX_CMD $INPUT_SCRIPT $INPUT_ARGVS
