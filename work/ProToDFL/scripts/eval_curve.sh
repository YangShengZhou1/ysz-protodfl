#!/bin/bash
# RUN-027 评估脚本：4 条轨迹 × 11 个快照 = 44 份，四卡并行
# 调用方式与 11-07 的 _remote_run_next_analysis.sh 完全一致，只改卡号、遍历轮次与输出目录
set -u
source /home/seczone/anaconda3/etc/profile.d/conda.sh
conda activate base
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/ysz" || exit 1

PY=/home/seczone/anaconda3/bin/python
DATA=/home/seczone/ysz/datasets/cifar10
OUT=outputs/experiments/2026-11-10_curve
ROUNDS="1 50 100 150 200 250 300 350 400 450 500"
mkdir -p "$OUT"

eval_one() {
  local tag=$1 gpu=$2
  local base
  base=$(ls -d outputs/checkpoints/*"$tag"* 2>/dev/null | head -1)
  if [ -z "$base" ]; then
    echo "[skip] $tag 无快照目录"
    return
  fi
  for r in $ROUNDS; do
    local ck="$base/round_$r"
    if [ ! -d "$ck" ]; then
      echo "[skip] $tag r$r 未落盘"
      continue
    fi
    echo "[start] $tag r$r gpu$gpu $(date +%T)"
    "$PY" scripts/kl_benefit_distribution.py \
      --checkpoint "$ck" --data_dir "$DATA" \
      --output "$OUT/kl_${tag}_r${r}.json" \
      --device "cuda:${gpu}" --scope base --max_candidates 20 --dataset cifar10 \
      > "$OUT/kl_${tag}_r${r}.log" 2>&1
    echo "[done] $tag r$r rc=$? $(date +%T)"
  done
}

eval_one cur-dir-s1 0 &
eval_one cur-dir-s2 1 &
eval_one cur-path-s1 2 &
eval_one cur-path-s2 3 &
wait
echo "CURVE_EVAL_DONE $(date +%T)"
ls -l "$OUT"/*.json 2>/dev/null | wc -l
