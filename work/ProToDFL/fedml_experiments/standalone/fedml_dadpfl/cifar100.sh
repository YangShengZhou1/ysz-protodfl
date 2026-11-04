#!/bin/bash
python main_dadpfl.py --model 'resnet18' \
--dataset 'cifar100' \
--partition_method 'my_part' \
--partition_alpha '0.1' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--prune_scale 1 \
--rigl \
--comm_round 501 \
--dense_ratio 0.5 \
--anneal_factor 0.3 \
--target_sparsity 0.8 \
--seed 2022 \
--regrow \
--reconfig_reduce 1.3 \
--early_prune_threshold 0.02 \
--different_initial \
--new_schedule \
--rr_interval 1 \
--max_wait 10 \
--cs 'random' \
--gpu 0 \
--dynamic 0 \
--tqdm \
--times 1

#!/bin/bash
python main_dadpfl.py --model 'resnet18' \
--dataset 'cifar100' \
--partition_method 'n_cls' \
--partition_alpha '30' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--prune_scale 1 \
--rigl \
--comm_round 501 \
--dense_ratio 0.5 \
--anneal_factor 0.3 \
--target_sparsity 0.8 \
--seed 2022 \
--regrow \
--reconfig_reduce 1.3 \
--early_prune_threshold 0.02 \
--different_initial \
--new_schedule \
--rr_interval 1 \
--max_wait 10 \
--cs 'random' \
--gpu 0 \
--dynamic 0 \
--tqdm \
--times 1

