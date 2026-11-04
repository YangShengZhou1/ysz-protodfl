#!/bin/bash
python main_dfedsam.py --model 'resnet18' \
--dataset 'cifar110' \
--partition_method 'n_cls' \
--partition_alpha 22 \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--rho 0.01 \
--cs 'random' \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--seed 2022 \
--tqdm \
--gpu 1

