#!/bin/bash
set -u
cd /home/seczone/ysz
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
export CUDA_VISIBLE_DEVICES=0
PY=/home/seczone/anaconda3/bin/python
CLIENTS="0,1,2,3,4,6,7,16,17,19"
mkdir -p outputs/checks/server-paired-final

for tag in srv20-e5-r10-all-s0 srv20-e5-r10-base-s0
do
  ckpt=$(find outputs/checkpoints -maxdepth 3 -type d -path "*${tag}/round_10" | head -1)
  if [ -z "$ckpt" ]; then
    echo "MISSING ${tag}"
    continue
  fi
  echo "CHECK ${tag} ${ckpt}"
  "$PY" scripts/paired_aggregation_check.py \
    --checkpoint "$ckpt" \
    --data_dir datasets/cifar10 \
    --output "outputs/checks/server-paired-final/${tag}" \
    --clients "$CLIENTS" --device cuda:0
done
