#!/bin/bash
set -uo pipefail
cd /home/seczone/ysz
mkdir -p outputs/console
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
ENTRY=fedml_experiments/standalone/ProToDFL/main_ProToDFL.py
COMMON=(
  --dataset cifar10 --model resnet18
  --data_dir datasets --output_dir outputs
  --partition_method dir --partition_alpha 0.3
  --client_num_in_total 20 --frac 1.0
  --comm_round 2 --epochs 5 --batch_size 256
  --cs ring --dynamic 0 --k 2 --t_max 100
  --momentums 0.5 --seed 0 --times 1 --step 0
  --max_neighbors 2 --validation_ratio 0.2 --save_models_round 1
)

timeout 7200 "$PY" "$ENTRY" "${COMMON[@]}" --gpu 0 --aggregate_scope all \
  --tag srv20-e5-all-s0 > outputs/console/srv20-e5-all-s0.log 2>&1 &
p0=$!
timeout 7200 "$PY" "$ENTRY" "${COMMON[@]}" --gpu 1 --aggregate_scope base \
  --tag srv20-e5-base-s0 > outputs/console/srv20-e5-base-s0.log 2>&1 &
p1=$!

wait "$p0"; r0=$?
wait "$p1"; r1=$?
echo "all_exit=$r0 base_exit=$r1"
exit $(( r0 != 0 || r1 != 0 ))
