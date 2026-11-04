#!/bin/bash
python main_gossipfl.py --model 'resnet18' \
--dataset 'tiny' \
--partition_method 'my_part' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--cs 'exp' \
--lr_decay 0.998 \
--epochs 5 \
--target_sparsity 0.5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--gpu 1 \
--seed 2022
