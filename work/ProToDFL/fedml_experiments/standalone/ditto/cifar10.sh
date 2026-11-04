#!/bin/bash
python main_ditto.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'n_cls' \
--partition_alpha 2 \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 501 \
--seed 2022 \
--epochs 2 \
--lamda 0.5 \
--local_epochs 3 \
--gpu 2 \
--dynamic 1