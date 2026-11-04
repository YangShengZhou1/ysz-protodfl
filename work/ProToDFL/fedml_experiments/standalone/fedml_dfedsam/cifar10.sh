#!/bin/bash
python main_dfedsam.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'my_part' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--cs 'random' \
--rho 0.01 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--seed 2022 \
--tqdm \
--gpu 0
