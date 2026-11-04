import copy
import logging
import time

import numpy as np
import pdb
import torch
from torch import nn

from fedml_api.model.cv.cnn_meta import Meta_net

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):#稀疏模型类，用于初始化稀疏模型，训练稀疏模型以及优化稀疏网络
    def __init__(self, model, args=None, logger = None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def init_masks(self, params, sparsities):
        masks ={}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))# 计算掩码中需要保留的非零元素数量
            if dense_numel > 0:
                temp = masks[name].view(-1)# 创建一个临时张量，将掩码展平为一维数组
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]# 从随机排列后的索引中选择前 dense_numel 个，然后将这些索引对应的元素设置为1，表示要保留
                temp[perm] =1
        return masks

    #根据不同的分布（均匀分布或ERK分布）和给定的稀疏度参数，在模型的不同层中计算和设置稀疏度
    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse = 0.5):#稀疏度0.5
        spasities = {}# 创建一个字典来存储计算后的稀疏度
        if distribution == "uniform": # 如果分布是均匀分布，那么所有层的稀疏度都将是相同的（1 - dense_ratio）
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.args.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":#为参数较多的层分配较高的稀疏度，为参数较少的层分配较低的稀疏度
            self.logger.info('initialize by ERK')
            total_params = 0# 统计所有模型参数的总数量
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()# 创建密集层

            density = sparse# 初始稀疏度
            while not is_epsilon_valid:
            # 我们将从所有层开始，尝试找到正确的 epsilon。但是，如果任何概率超过 1，
            # 我们将使该层变得密集，并在非密集层中重复这个过程（查找 epsilon）。
            # 我们希望总连接数保持不变。假设我们有四个层，分别有 N_1, N_2, N_3, N_4 个参数。
            # 假设经过一些迭代后，某些密集层（3, 4）的概率超过了 1，因此我们将它们添加到 dense_layers 集合中。
            # 这些层将不会按 erdos_renyi 进行缩放，但我们需要计数它们，以便达到目标参数计数。
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
            
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:#根据 dense_layers 中是否包含当前参数的名称，来更新 rhs。
                        rhs -= n_zeros#如果 name 在 dense_layers 中，表示这是一个密集层，所以减去 n_zeros。
                    else:
                        rhs += n_ones# 计算原始概率，并应用幂次方尺度
                        raw_probabilities[name] = (
                                                     np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  ) ** self.args.erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:#如果某个层的概率超过1，则将该层标记为密集层，并重新计算 epsilon
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0#密集层，那么将其稀疏度 spasities[name] 设置为0。这意味着密集层的所有参数都将被保留，不会被剪枝，稀疏度为0表示完全不稀疏
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])#其中 epsilon 控制了整体稀疏度的程度，raw_probabilities[name] 表示该层的原始概率值。epsilon 乘以 raw_probabilities[name] 的结果表示了该层的稀疏度，数值在0到1之间，越接近0表示越稀疏，越接近1表示越密集（1-）
        return spasities

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def screen_gradients(self, train_data, device):#通过训练数据存储模型更新梯度
        model = self.model
        model.to(device)
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(train_data))
        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}
        for name, param in model.named_parameters():
            gradient[name] = param.grad.to("cpu")
        return gradient


    def train(self, train_data,  device,  args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)#初始化SGD优化器
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)#对模型的梯度进行裁剪，裁剪的过程会计算参数梯度的L2范数，如果范数大于指定的阈值（10），那么梯度将被重新缩放，以使其范数等于阈值。这有助于避免梯度爆炸，使训练过程更加稳定
                optimizer.step()
                epoch_loss.append(loss.item())
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        param.data *= self.masks[name].to(device)#将模型参数中与掩码中对应位置的元素相乘，以实现参数剪枝（pruning）。这是一种在稀疏神经网络中常见的技术，用于根据掩码来控制哪些参数需要被保留，哪些需要被设置为零，从而实现模型的稀疏化。
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))



    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

