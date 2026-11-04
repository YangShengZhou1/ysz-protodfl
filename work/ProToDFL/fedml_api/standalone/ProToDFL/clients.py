import copy
import logging
import math
import time
import pdb
import numpy as np
import torch
import torch.nn.functional as F
class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger,w, local_val_data=None, local_val_num=0):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_validation_data = local_val_data
        self.local_validation_num = local_val_num
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.momentum = args.momentums
        self.global_mean = None
        self.selected_neighbors = set()
        self.selected_weigh = {}
        self.neighbor_dict = []
        self.neigh_distance = {}
        self.second_neighbors = {}
        self.threshold = 500
        self.thres_id = 0
        self.num = 0
        self.max_num = 0
        self.siganl = 0
        self.wait_signal = 0#0表示需要二阶邻居节点优化拓扑网络，1表示处于等待期，2表示刚结束等待期（临界状态）
        self.wait_time = [0,0]#wait_time[0]记录还剩下多少等待时间，wait_time[1]记录总共需要等待的时间
        self.count_num = 0
        
        
        self.model_trainer.set_model_params(w)
        rep = self.model_trainer.train_rep(self.local_training_data, self.device, self.args)
        self.running_mean = torch.zeros_like(rep[0])

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number
    
    def reset_running_stats(self):
        self.running_mean.zero_()  
        #self.num_batches_tracked.zero_()

    def train(self, w_global,round):# -> tuple:# -> tuple:# -> tuple:# -> tuple:
        # self.logger.info(sum([torch.sum(w_per[name]) for name in w_per]))
        num_comm_params = self.model_trainer.count_communication_params(w_global)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        if round == 0:
          self.reset_running_stats()
        running_mean = self.model_trainer.train(self.local_training_data, self.device, self.args, round, copy.deepcopy(self.running_mean), self.momentum)
        if round == 0 or getattr(self.args, 'proto_update_each', 0) == 1:
            self.running_mean = running_mean
        weights = self.model_trainer.get_model_params(self.device)
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("################client:{} threshold : {}".format(self.client_idx,self.threshold))
            
            
        self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        # self.logger.info( "training_flops{}".format( self.model_trainer.count_training_flops_per_sample()))
        # self.logger.info("full{}".format(self.model_trainer.count_full_flops_per_sample()))
        self.logger.info("--------------------------------------")
        #training_flops = self.args.epochs * self.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        num_comm_params += self.model_trainer.count_communication_params(weights)
        return  weights, tst_results, num_comm_params

    def local_test(self, w, b_use_test_dataset = True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
    def local_val(self, w):
        test_data = self.local_val_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
