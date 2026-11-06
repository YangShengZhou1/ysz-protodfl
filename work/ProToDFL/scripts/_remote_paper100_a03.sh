#!/bin/bash
# 论文级完整复现：CIFAR-10 / Dirichlet alpha=0.3 / 100 客户端 / my_part 10 组 /
# random 10 邻居 / 500 轮 / 5 local epoch / batch 128 / lr 0.1 不衰减 / 只聚合特征提取器
# 快照点 r1(对接旧结果) r250 r500(论文终点)，3 x 4.3G ≈ 12.9G
set -u
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
ENTRY=fedml_experiments/standalone/ProToDFL/main_ProToDFL.py
mkdir -p outputs/console
TAG=paper100-a03-r500

CUDA_VISIBLE_DEVICES=1 setsid nohup timeout 129600 "$PY" "$ENTRY" \
  --dataset cifar10 --model resnet18 \
  --data_dir datasets --output_dir outputs \
  --partition_method my_part --partition_alpha 0.3 \
  --client_num_in_total 100 --frac 1.0 \
  --comm_round 500 --epochs 5 --batch_size 128 \
  --lr 0.1 --lr_decay 1.0 \
  --cs random --dynamic 0 --k 2 --t_max 100 \
  --times 1 --step 0 \
  --max_neighbors 0 --validation_ratio 0.2 \
  --save_models_round 1,250,500 \
  --gpu 0 --seed 0 --aggregate_scope base --tag "$TAG" \
  > "outputs/console/${TAG}.log" 2>&1 &

sleep 3
echo "started $TAG"
