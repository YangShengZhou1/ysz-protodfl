import copy
import logging
import math
import torch
from torch import nn
import numpy as np
import pdb
import torch
from torch.autograd import Variable
class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger,w):#初始化的时候最后要加上初始的权重参数
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.grad_accumulator = {}
        self.klw = args.kl_weight
        self.momentum = args.momentums
        self.global_mean = None
        self.selected_neighbors = set()
        self.selected_weigh = {}
        self.neighbor_dict = []
        self.neigh_distance = {}
        self.second_neighbors = {}
        self.threshold = 50
        self.max_num = 0
        self.siganl = 0
        
        self.model_trainer.set_model_params(w)
        rep = self.model_trainer.train_rep(self.local_training_data, self.device, self.args)
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)
        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        #self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number
    
    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def train(self,w,round):
        # 计算通信参数的数量
        num_comm_params = self.model_trainer.count_communication_params(w)# 设置模型参数为传入的 w
        self.model_trainer.set_model_params(w)#设置模型初始化参数w^0
        self.model_trainer.set_id(self.client_idx)# 在本地测试数据上进行测试，并记录测试结果（先测试训练前的模型效果）
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args,self.client_mean)
        
        self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        # 在本地训练数据上进行训练
        self.reset_running_stats()
        running_mean,client_mean = self.model_trainer.train(self.local_training_data, self.device, self.args, self.running_mean, self.global_mean, self.client_mean, self.momentum, self.klw, round)#train_data, device,  args, old_running_mean, global_mean, client_mean, momentum, round
        self.running_mean,self.client_mean = running_mean,client_mean
        weights = self.model_trainer.get_model_params()# 获取训练后的模型参数
        self.model_trainer.set_model_params(weights)# 将模型参数设置为训练后的weights
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args, self.client_mean)# 在本地测试数据上进行测试，并记录测试结果
        self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))

        update = {}# 计算权重更新
        for name in weights:
            update[name] = weights[name] - w[name]

        self.logger.info("-----------------------------------")
        num_comm_params += self.model_trainer.count_communication_params(update)
        return weights, update, num_comm_params, tst_results

    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args, self.client_mean)
        return metrics
    
    

                
        
    
            
            