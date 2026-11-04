import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from fedml_api.standalone.fedavg.client import Client
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon, cosine
from torch import nn
from fedml_api.utils.NetworkTopology import NetworkTopology

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        self.cs = args.cs
        self.graph = self.generate_graph()
        self.sub = 0
        self.add = 0
        w_global = self.model_trainer.get_trainable_params(device)
        self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
                    
    def compare_exper(self,client):
        selected_neighbors = set()
        for neigh_id in self.graph.adj[client.client_idx]:
            selected_neighbors.add(neigh_id)
        client.selected_neighbors = selected_neighbors

        
    def generate_graph(self):
        network_topology = NetworkTopology(self.args.client_num_in_total)

        if self.args.cs == 'ring':
            graph = network_topology.generate_graph('ring')
        elif self.args.cs == 'grid':
            graph = network_topology.generate_graph('grid')
        elif self.args.cs == 'exp':
            graph = network_topology.generate_graph('exp')
        elif self.args.cs == 'full':
            graph = network_topology.generate_graph('full')
        elif self.args.cs == 'random':
            graph = network_topology.generate_graph('random')
        else:
            while True:
                graph = nx.random_geometric_graph(self.args.client_num_in_total, 0.25)
                if nx.is_k_edge_connected(graph, 1):
                    break
        return graph

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        w_global = self.model_trainer.get_trainable_params(self.device)
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger,w_global)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        num_comm_params = {}
        for client in self.client_list:#初始化
            client.max_num = len(self.graph.adj[client.client_idx])
        #w_global = self.model_trainer.get_model_params()
        #self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            #weights, training_flops,_ = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
            self.compare_exper(c)#w_per_globals[c.client_idx] = copy.deepcopy(weights)
            #num_comm_params[c.client_idx] = num_comm_params
            #c.neighbor_dict = self._benefit_choose(c.client_idx, self.args.client_num_in_total,
            #                                      self.args.client_num_per_round, self.args.cs)
        #self.graph = self.generate_graph()
        # device = {device} cuda:0apply mask to init weights
        ##########################################################################################################
        ###################################****协议全局训练开始****#################################################
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
            if self.args.dynamic == 1:#dynamic topology
                if round_idx % 10 == 0:#per 10 rounds change topology
                    self.graph = self.generate_graph()
            for c in self.client_list:
                self.compare_exper(c)
            self.logger.info("################Communication round : {}".format(round_idx))
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            w_locals = []
            self.logger.info('add number:{}'.format(self.add))
            self.logger.info('sub number:{}'.format(self.sub))
            
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
    
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, str(c.client_idx)))#该客户端在本轮训练中活跃，记录训练日志
                
                #客户端c选择邻居客户端节点
                self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
            self._aggregate_func()#聚合模型
            for c in self.client_list:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))
                # update dataset
                # update meta components in personal network
                weights, update, tst_results = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, round_idx)
                self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
                #num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)

            self._test_on_all_clients(round_idx)
            # self._local_test_on_all_clients(w_global, round_idx)
        # self.record_avg_inference_flops(w_global)


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _test_on_all_clients(self, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(copy.deepcopy(self.w_per_globals[client_idx]), True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))

            #p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            #p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            #p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            #p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        # test on test dataset
        g_test_acc = sum([np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        #p_test_acc = sum(
        #    [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
        #     range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        #p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
        #                   range(self.args.client_num_in_total)]) / self.args.client_num_in_total


        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.logger.info(stats)
        
        

        #stats = {'person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        #self.stat_info["person_test_acc"].append(p_test_acc)
        #self.logger.info(stats)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
        
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
    
    
    def _aggregate_func(self):#FedAvg平均聚合
        aggregate_weight = {}
        for c in self.client_list:
            aggregate_weight.update({c.client_idx : {k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}})
        for c in self.client_list:
            total_num = 0
            #global_mean = 0
            total_num += self.train_data_local_num_dict[c.client_idx]
            for neigh_index in c.selected_neighbors:
                total_num += self.train_data_local_num_dict[neigh_index]
            W = self.train_data_local_num_dict[c.client_idx] / total_num
            #global_mean += all_running_means[c.client_idx] * W #初始化聚合模型和global_mean
            for k in aggregate_weight[c.client_idx].keys():
                aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]#初始化
            for neigh_index in c.selected_neighbors:
                w = self.train_data_local_num_dict[neigh_index] / total_num
                for k in aggregate_weight[c.client_idx].keys():
                    aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
                #global_mean += all_running_means[neigh_index] * w
            #c.global_mean = global_mean
        for c in self.client_list:    
            self.w_per_globals[c.client_idx] = aggregate_weight[c.client_idx]   
        
        self.logger.info('Doing local aggregation!')
    
    