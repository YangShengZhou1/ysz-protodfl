import copy
import logging
import math
import pickle
import random
import time
from tqdm import tqdm
import pdb
import numpy as np
import torch
from torch import nn
import networkx as nx

from fedml_api.standalone.DBE.client import Client


class DBE_API(object):
    def __init__(self, dataset, device, args, model_trainer, logger):#model_trainer即管理稀疏神经网络的类，其中包含对稀疏神经网络的各项操作
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_counts = class_counts
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.sub = 0
        self.add = 0
        #self.w_per_globals = []
        self.graph = self.generate_graph()
        w_global = self.model_trainer.get_model_params()
        self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
            self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
        all_running_means = {client.client_idx : client.running_mean for client in self.client_list}
        self._aggregate_func(all_running_means,begin = 0)
        
        
        
        
    def generate_graph(self):
            # time-varying topology
            while True:
                graph = nx.random_geometric_graph(self.args.client_num_in_total,0.25)
                if nx.is_k_edge_connected(graph, 1)==True:
                    break
            return graph
        
    def remove_connection(self,graph, client_id1, client_id2):
            """
            Removes a connection (edge) between two clients in the network graph.

            Parameters:
            - graph: A networkx graph object representing the network topology.
            - client_id1: ID of the first client.
            - client_id2: ID of the second client.
            """
            if graph.has_edge(client_id1, client_id2):
                graph.remove_edge(client_id1, client_id2)
                self.sub += 1
                #print(f"Connection removed between Client {client_id1} and Client {client_id2}.")
            #else:
                #print(f"No existing connection to remove between Client {client_id1} and Client {client_id2}.")
        
    def add_connection(self,graph, client_id1, client_id2):
            """
            Adds a connection (edge) between two clients in the network graph. If the connection
            already exists, reports that the connection already exists.

            Parameters:
            - graph: A networkx graph object representing the network topology.
            - client_id1: ID of the first client.
            - client_id2: ID of the second client.
            """
            if client_id1 in graph.nodes() and client_id2 in graph.nodes():
                if graph.has_edge(client_id1, client_id2):
                    print(f"Connection already exists between Client {client_id1} and Client {client_id2}.")
                else:
                    graph.add_edge(client_id1, client_id2)
                    self.add += 1

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        #self.generate_connection_matrix()# 保存dist矩阵，存储每个client和其他client的距离
        w_global = self.model_trainer.get_model_params()
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger, w_global)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        # for first initialization
        params = self.model_trainer.get_trainable_params()
        
        #w_global = self.model_trainer.get_model_params()
        num_comm_params = {}
        #self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        ##########################################################################################################
        ###################################模型初始化：w_global和mask##############################################
        ##########################################################################################################
        
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
            self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
            num_comm_params[c.client_idx] = num_comm_param
            #c.neighbor_dict = self._benefit_choose(c.client_idx, self.args.client_num_in_total,
            #                                      self.args.client_num_per_round, self.args.cs)#cur_clnt, client_num_in_total, client_num_per_round, cs = False
            #在全局初始化阶段，每个本地客户端先进行K轮本地训练获得本地模型的更新梯度、以及loss损失
        print("num_comm_params:")
        print(len(num_comm_params))
        print(num_comm_params)
        self.graph = self.generate_graph()
        
        ##########################################################################################################
        ###################################Dis-pFL协议全局训练开始#################################################
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
            self.logger.info("################Communication round : {}".format(round_idx))
            #active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])#根据概率分布随机选择哪些客户端在当前通信轮次中活跃（1活跃，0表示不活跃）
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            #对每个客户端进行操作
            all_running_means = {client.client_idx : client.running_mean for client in self.client_list}
            
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))#该客户端在本轮训练中活跃，记录训练日志
                
                #客户端c选择邻居客户端节点
                if self.args.cs!="full":
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
                else:
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
                
            self._aggregate_func(all_running_means)#聚合模型
            for c in self.client_list:# 聚合好模型后，设置client进行local training
                weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
                self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
                num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)
            #for c in self.client_list:
            #    c.neighbor_dict = self._benefit_choose(c.client_idx, self.args.client_num_in_total,
            #                                      self.args.client_num_per_round, self.args.cs)
            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
        #完成所有通信轮次后
        return
    
    
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _benefit_choose(self, cur_clnt, client_num_in_total, client_num_per_round, cs = False):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd==1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes==cur_clnt)[0]))
        return client_indexes

        

    def _aggregate_func(self,all_running_means,begin = 1):
        if begin == 0:
            for c in self.client_list:
                total_num = 0
                global_mean = 0
                total_num += self.train_data_local_num_dict[c.client_idx]
                for neigh_index in self.graph.adj[c.client_idx]:
                    total_num += self.train_data_local_num_dict[neigh_index]
                W = self.train_data_local_num_dict[c.client_idx] / total_num
                global_mean += all_running_means[c.client_idx] * W#初始化聚合模型和global_mean
                for neigh_index in self.graph.adj[c.client_idx]:
                    w = self.train_data_local_num_dict[neigh_index] / total_num
                    global_mean += all_running_means[neigh_index] * w
                c.global_mean = global_mean
        else:
            aggregate_weight = {}
            for c in self.client_list:
                aggregate_weight.update({c.client_idx : copy.deepcopy(self.w_per_globals[c.client_idx])})
            for c in self.client_list:
                total_num = 0
                global_mean = 0
                total_num += self.train_data_local_num_dict[c.client_idx]
                for neigh_index in self.graph.adj[c.client_idx]:
                    total_num += self.train_data_local_num_dict[neigh_index]
                W = self.train_data_local_num_dict[c.client_idx] / total_num
                global_mean += all_running_means[c.client_idx] * W#初始化聚合模型和global_mean
                for k in aggregate_weight[c.client_idx].keys():
                    aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]#初始化
                for neigh_index in self.graph.adj[c.client_idx]:
                    w = self.train_data_local_num_dict[neigh_index] / total_num
                    for k in aggregate_weight[c.client_idx].keys():
                        aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
                    global_mean += all_running_means[neigh_index] * w
                c.global_mean = global_mean
            for c in self.client_list:#全部聚合完成后，更新全局维护的w_per_globals表    
                self.w_per_globals[c.client_idx] = aggregate_weight[c.client_idx]   
            
            self.logger.info('Doing local aggregation!')
        
        
    

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
        self.logger.info("################local_test_on_all_clients after local training in communication round: {}".format(round_idx))
        test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }
        for client_idx in range(self.args.client_num_in_total):
                # test data
                test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(self.args.client_num_in_total) ] )/self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)])/self.args.client_num_in_total

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, round_idx):
        self.logger.info("################local_test_on_all_clients before local training in communication round: {}".format(round_idx))
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.args.client_num_in_total):

            # test data
            test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["new_mask_test_acc"].append(test_acc)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops=[]
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers== None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] *mask_pers[client_idx][name]
                inference_flops+= [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops)/len(inference_flops)
        self.stat_info["avg_inference_flops"]=avg_inference_flops
        
    

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["label_num"] =self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []
        



