#!/usr/bin/env bash
set -euo pipefail

QUESTION_IDS=(1 3 10 43 46 59 69 74 86 91 92 95 100 103 111 120 125 127 129)

for qid in "${QUESTION_IDS[@]}"; do
  data_dir="lcb_v6_singles/q_${qid}"
  run_name="multiturn_q${qid}"

  echo "Running qid=${qid}"
  python baseline_multiturn/multiturn.py \
    --data-dir="$data_dir" \
    --run-name="$run_name" \
    --seed=0
done