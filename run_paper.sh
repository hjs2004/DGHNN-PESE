#!/usr/bin/env bash
# Run all 30 main-model experiments on the selected GPUs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gpu_list="${1:-0,1,2,3,4,5,6,7}"
python_bin=/root/venvs/dghnn/bin/python
mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/paper-$(date +%Y%m%d-%H%M%S)-XXXXXX")
nohup "$python_bin" -u parallel_experiments.py --gpus "$gpu_list" \
    --indices 0,1,2,3,4,5 --seeds 0,1,2,3,4 --epochs 3000 \
    --run-dir "$run_dir" > "$run_dir/scheduler.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/launcher.pid"
printf '%s\n' "$run_dir" > last_paper_run.txt
printf 'Submitted scheduler PID=%s\nRun: %s\n' "$!" "$run_dir"
printf 'Status: %s parallel_experiments.py --status %q\n' "$python_bin" "$run_dir"
printf 'Stop: %s parallel_experiments.py --stop %q\n' "$python_bin" "$run_dir"
