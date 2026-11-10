#!/bin/bash
# RUN-027 启动脚本：两格各补 seed1、seed2，四卡并行，11 个快照点
set -u
source /home/seczone/anaconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/ysz" || exit 1

BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{c++} END{print c+0}')
if [ "$BUSY" -gt 0 ]; then
  echo "GPU 非空闲（$BUSY 张占用超过 500MiB），退出"
  exit 1
fi
if pgrep -f 'main_ProToDFL.py' > /dev/null; then
  echo "已有 main_ProToDFL 进程，退出"
  exit 1
fi

R=1,50,100,150,200,250,300,350,400,450,500
C="--dataset cifar10 --model resnet18 --data_dir datasets --output_dir outputs --client_num_in_total 100 --frac 1.0 --comm_round 500 --epochs 5 --batch_size 128 --lr 0.1 --lr_decay 1.0 --wd 0.0005 --client_optimizer sgd --momentum 0 --cs random --k 2 --t_max 100 --dynamic 0 --step 0 --max_neighbors 0 --momentums 0.1 --proto_update_each 0 --aggregate_scope base --self_weight_scale 1.0 --oracle_m 2 --oracle_mode 0 --personal 0 --times 1 --validation_ratio 0.2 --frequency_of_the_test 1 --ci 0"

launch() {
  local name=$1
  shift
  nohup setsid python fedml_experiments/standalone/ProToDFL/main_ProToDFL.py $C "$@" \
    --save_models_round $R --tag "$name" \
    > "outputs/console/$name.log" 2>&1 < /dev/null &
  echo "started $name pid $!"
}

launch cur-dir-s1  --partition_method my_part --partition_alpha 0.3 --seed 1 --gpu 0
launch cur-dir-s2  --partition_method my_part --partition_alpha 0.3 --seed 2 --gpu 1
launch cur-path-s1 --partition_method n_cls  --partition_alpha 2.0 --seed 1 --gpu 2
launch cur-path-s2 --partition_method n_cls  --partition_alpha 2.0 --seed 2 --gpu 3

sleep 8
echo "=== 进程 ==="
pgrep -af 'main_ProToDFL.py' | sed 's/ -.*--tag/ --tag/' | head -6
echo "=== 磁盘 ==="
df -h /home | tail -1
