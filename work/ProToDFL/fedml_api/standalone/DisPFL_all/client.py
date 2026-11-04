import copy
import logging
import math

import numpy as np
import pdb
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w, masks,round):
        # 计算通信参数的数量
        num_comm_params = self.model_trainer.count_communication_params(w)# 设置模型参数为传入的 w
        self.model_trainer.set_model_params(w)# 设置掩码为传入的 mask
        self.model_trainer.set_masks(masks)# 设置客户端的标识
        self.model_trainer.set_id(self.client_idx)# 在本地测试数据上进行测试，并记录测试结果（先测试训练前的模型效果）
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        # 在本地训练数据上进行训练
        self.model_trainer.train(self.local_training_data, self.device, self.args, round)
        weights = self.model_trainer.get_model_params()# 获取训练后的模型参数
        self.model_trainer.set_model_params(weights)# 将模型参数设置为训练后的 weights
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)# 在本地测试数据上进行测试，并记录测试结果
        self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))

        update = {}# 计算权重更新
        for name in weights:
            update[name] = weights[name] - w[name]

        self.logger.info("-----------------------------------")
        gradient = None
        if not self.args.static:# 如果不是静态模型
            if not self.args.dis_gradient_check:# 如果不进行梯度检查
                gradient = self.model_trainer.screen_gradients(self.local_training_data, self.device)# 计算梯度
            masks, num_remove = self.fire_mask(masks, weights, round)#对稀疏网络的mask网络进行dropout移除多余的网络参数模型
            masks = self.regrow_mask(masks, num_remove, gradient)# 对 masks 进行再生长操作，根据梯度重新添加网络参数模型（个性化）
        sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()# 计算每个数据样本的稀疏训练计算量
        full_flops = self.model_trainer.count_full_flops_per_sample()# 计算完全密集模型的训练计算量
        self.logger.info("training flops per data {}".format(sparse_flops_per_data))
        self.logger.info("full flops for search {}".format(full_flops))
        # we train the data for `self.args.epochs` epochs, and forward one epoch of data with full density to screen gradient.
        training_flops = self.args.epochs*self.local_sample_number*sparse_flops_per_data+\
                         self.args.batch_size* full_flops# 计算训练的总计算量

        # uplink params # 更新通信参数的数量
        num_comm_params += self.model_trainer.count_communication_params(update)
        return masks,  weights, update, training_flops, num_comm_params, tst_results


    def fire_mask(self, masks, weights, round):
        # 计算丢弃比例（类似dropout操作）
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.comm_round))
        
        new_masks = copy.deepcopy(masks)
        # 创建一个字典来存储每个层的要移除的元素数量
        num_remove = {}
        # 遍历 masks 中的每个层
        for name in masks:
            # 计算当前层中非零元素的数量
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)# 根据 drop_ratio 计算要移除的元素数量（向上取整）
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]), 100000 * torch.ones_like(weights[name]))# 创建一个临时权重张量，其中只保留了 masks 中大于零的部分，并将小于零的部分设置为一个大正数
            x, idx = torch.sort(temp_weights.view(-1).to(self.device))# 对临时权重张量进行排序，并获取要移除的最重要的 num_remove[name] 个元素的索引
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0# 在 new_masks 中将这些索引对应的元素设置为零，表示要移除
        return new_masks, num_remove# 返回新的 masks 和 num_remove 字典


    # we only update the private components of client's mask
    def regrow_mask(self, masks,  num_remove, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            # if name not in public_layers:
                # if "conv" in name:
                if not self.args.dis_gradient_check:
                    # 如果不进行梯度检查，计算一个临时张量 temp，其中对于 masks 中的值为零的位置，取 gradient 中对应位置的绝对值，
                    # 否则将这些位置设置为一个极小的负数（-100000）
                    temp = torch.where(masks[name] == 0, torch.abs(gradient[name]), -100000 * torch.ones_like(gradient[name]))
                    sort_temp, idx = torch.sort(temp.view(-1).to(self.device), descending=True)# 对 temp 进行降序排序，并获取前 num_remove[name] 个元素的索引
                    new_masks[name].view(-1)[idx[:num_remove[name]]] = 1# 在 new_masks 中将这些索引对应的元素设置为1，表示要保留
                else:
                    temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )# 如果进行梯度检查，创建一个临时张量 temp，其中对于 masks 中的值为零的位置，设置为1，否则设置为0
                    idx = torch.multinomial( temp.flatten().to(self.device),num_remove[name], replacement=False)
                    new_masks[name].view(-1)[idx]=1# 从 temp 中多项式采样出 num_remove[name] 个不重复的索引，并将这些索引对应的元素设置为1，表示要保留
        return new_masks


    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
