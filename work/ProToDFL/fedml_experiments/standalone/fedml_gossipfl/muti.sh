#!/bin/bash
python main_gossipfl.py --model 'resnet18' \
--dataset 'cifar110' \
--partition_method 'n_cls' \
--partition_alpha 22 \
--batch_size 128 \
--lr 0.1 \
--cs 'exp' \
--lr_decay 0.998 \
--epochs 5 \
--target_sparsity 0.5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--gpu 0 \
--seed 2022 
