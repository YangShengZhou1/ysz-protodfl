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
  --comm_round 10 --epochs 5 --batch_size 256
  --cs ring --dynamic 0 --k 2 --t_max 100
  --momentums 0.5 --times 1 --step 0
  --max_neighbors 2 --validation_ratio 0.2 --save_models_round 10
)

run_one () {
  local gpu=$1 seed=$2 scope=$3 tag=$4
  CUDA_VISIBLE_DEVICES="$gpu" timeout 14400 "$PY" "$ENTRY" "${COMMON[@]}" \
    --gpu 0 --seed "$seed" --aggregate_scope "$scope" --tag "$tag" \
    > "outputs/console/${tag}.log" 2>&1 &
}

run_one 0 0 all  srv20-e5-r10-all-s0
run_one 1 0 base srv20-e5-r10-base-s0
run_one 2 1 all  srv20-e5-r10-all-s1
run_one 3 1 base srv20-e5-r10-base-s1
wait

run_one 0 2 all  srv20-e5-r10-all-s2
run_one 1 2 base srv20-e5-r10-base-s2
wait
