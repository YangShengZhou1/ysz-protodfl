#!/bin/bash
python FedAvg_FT_api.py --model 'resnet18' \
--dataset 'cifar100' \
--partition_method 'n_cls' \
--partition_alpha 3 \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--cs 'exp' \
--seed 2022 \
--gpu 0 \
--momentums 0.5 
#--device 'cuda' \
#--gpu 3
