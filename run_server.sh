#!/usr/bin/env bash
# Usage: bash run_server.sh 1 (physical GPU index from nvidia-smi)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gpu="${1:-1}"
python_bin="/root/venvs/dghnn/bin/python"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash run_server.sh GPU_INDEX" >&2
    exit 1
fi
free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
free_mib=${free_mib//[[:space:]]/}
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 20480 )); then
    echo "GPU $gpu has less than 20 GiB free; select another idle GPU with nvidia-smi." >&2
    exit 1
fi
test -x "$python_bin"
mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/server-$(date +%Y%m%d-%H%M%S)-XXXXXX")
export CUDA_VISIBLE_DEVICES="$gpu"
export DGHNN_PROFILE=server
export DGHNN_EDGE_CHUNK_SIZE="${DGHNN_EDGE_CHUNK_SIZE:-1048576}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export DGHNN_OUTPUT_DIR="$run_dir/outputs"
nohup "$python_bin" -u train.py > "$run_dir/train.log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$run_dir/train.pid"
printf '%s\n' "$train_pid" > train.pid
printf 'Submitted PID=%s, GPU=%s\nLog: %s/train.log\nOutputs: %s/outputs\n' "$train_pid" "$gpu" "$run_dir" "$run_dir"
printf 'Check: ps -p %s -o pid,etime,cmd\n' "$train_pid"
printf 'Follow: tail -f %q/train.log\n' "$run_dir"
