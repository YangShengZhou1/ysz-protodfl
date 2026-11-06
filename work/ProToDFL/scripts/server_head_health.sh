#!/bin/bash
set -u
cd /home/seczone/ysz
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
export CUDA_VISIBLE_DEVICES=0
PY=/home/seczone/anaconda3/bin/python
mkdir -p outputs/checks/server-paired

for tag in \
  srv20-e5-all-s0 srv20-e5-base-s0 \
  srv20-e5-all-s1 srv20-e5-base-s1 \
  srv20-e5-all-s2 srv20-e5-base-s2
do
  ckpt=$(find outputs/checkpoints -maxdepth 3 -type d -path "*${tag}/round_1" | head -1)
  out="outputs/checks/server-paired/${tag}"
  "$PY" scripts/paired_aggregation_check.py \
    --checkpoint "$ckpt" --data_dir datasets/cifar10 \
    --output "$out" --device cuda:0 > /dev/null
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); rows=d['rows']; print(sys.argv[2], 'clients', len(rows), 'pre_unique_minmax', min(r['pre_unique_predictions'] for r in rows), max(r['pre_unique_predictions'] for r in rows), 'base_unique_minmax', min(r['agg_base_unique_predictions'] for r in rows), max(r['agg_base_unique_predictions'] for r in rows))" "$out.json" "$tag"
done
