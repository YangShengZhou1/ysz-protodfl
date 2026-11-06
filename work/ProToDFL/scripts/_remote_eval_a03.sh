#!/bin/bash
# 对 paper100-a03-r500 的三个快照做全量候选评估。
# 结果只写 JSON/CSV（约 4MB），不产生新快照。
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
DATA=/home/seczone/ysz/datasets/cifar10
OUT=outputs/experiments/2026-11-06_paper100
BASE=$(ls -d outputs/checkpoints/*paper100* | head -1)
mkdir -p "$OUT"
echo "base: $BASE"
for r in 1 250 500; do
  echo "=== round_${r} ==="
  date +%T
  "$PY" scripts/kl_benefit_distribution.py \
    --checkpoint "$BASE/round_${r}" --data_dir "$DATA" \
    --output "$OUT/r${r}.json" \
    --device cuda:2 --scope base \
    --max_candidates 20 --dataset cifar10 \
    > "$OUT/r${r}.log" 2>&1
  echo "exit=$?"
  grep -o '"n_pairs": [0-9]*' "$OUT/r${r}.json" | head -1
  date +%T
done
echo '=== files ==='
ls -l "$OUT"/*.json
df -h / | tail -1
