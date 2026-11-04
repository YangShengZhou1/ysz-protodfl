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
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon, cosine


from fedml_api.standalone.Ours import client
from fedml_api.standalone.Ours.client import Client


class global_API(object):
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
                    
    def compare_exper(self,client):
        selected_neighbors = set()
        for neigh_id in self.graph.adj[client.client_idx]:
            selected_neighbors.add(neigh_id)
        client.selected_neighbors = selected_neighbors

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
        #params = self.model_trainer.get_trainable_params()
        #self.graph = self.generate_graph()
        #w_global = self.model_trainer.get_model_params()
        num_comm_params = {}
        for client in self.client_list:#初始化
            client.max_num = len(self.graph.adj[client.client_idx])
        #self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        # 初始化
        ##########################################################################################################
        ###################################模型初始化：w_global和mask##############################################
        ##########################################################################################################
        
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
            self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
            num_comm_params[c.client_idx] = num_comm_param
            #在全局初始化阶段，每个本地客户端先进行K轮本地训练获得本地模型的更新梯度、以及loss损失
        print("num_comm_params:")
        print(len(num_comm_params))
        print(num_comm_params)

        threshold = 2.4#逐步缩小阈值
        ##########################################################################################################
        ###################################****协议全局训练开始****#################################################
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
           
            
            self.logger.info("################Communication round : {}".format(round_idx))
            #active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])#根据概率分布随机选择哪些客户端在当前通信轮次中活跃（1活跃，0表示不活跃）
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            #对每个客户端进行操作
            all_running_means = {client.client_idx : client.running_mean for client in self.client_list}
            self.logger.info('add number:{}'.format(self.add))
            self.logger.info('sub number:{}'.format(self.sub))
            for client in self.client_list:#初始化
                client.signal = 0
                client.selected_weigh.clear()#选中节点的余弦距离
                client.selected_neighbors.clear()#选中节点集合
                client.second_neighbors.clear()#二阶邻居节点的余弦距离
                client.neigh_distance.clear()#一阶邻居节点的余弦距离
                client.threshold = 50
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))#该客户端在本轮训练中活跃，记录训练日志                
                self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
            for client in self.client_list:
                self.compare_exper(client)
            #    self.select_neighbors(client,similarity_threshold = threshold)
            #    #self.logger.info("-----------------------最初------------------")
            #    #self.logger.info('client_id: {}'.format(client.client_idx))
            #    #self.logger.info('client.neigh_distance: {}'.format(client.neigh_distance))
                #self.logger.info('client.selected_neighbors: {}'.format(client.selected_neighbors))
                #self.logger.info('client.signal: {}'.format(client.signal))
                #client.show_label()                   
            #for client in self.client_list:
            #    self.select_neighbors_two(client,similarity_threshold = threshold)
                
            #self.change_neighbors(grap=0,similarity_threshold = threshold)#cifar10 2.15 cifar100 1.2 fmnist 1.6
            self._aggregate_func(all_running_means)#聚合模型
            for c in self.client_list:# 聚合好模型后，设置client进行local training
                weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(self.w_per_globals[c.client_idx]), 0)
                self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
                num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)

            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
            
            threshold = max(threshold- 0.01 * round_idx , 0.80)
            #else:
            #    threshold = max(threshold- 0.01 * round_idx , 1.20)
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

    def _aggregate_func_cos(self,all_running_means,similarity_threshold = 2,begin = 1):#余弦距离倒数聚合模型权重，原型向量平均聚合
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
                global_mean = 0
                #total_num = 0
                #total_num += self.train_data_local_num_dict[c.client_idx]
                #for neigh_index in c.selected_neighbors:
                #    total_num += self.train_data_local_num_dict[neigh_index]
                close_neighbors = {k: v for k, v in c.selected_weigh.items() if v < similarity_threshold}
                if len(close_neighbors)>len(c.selected_neighbors)/2:#大于过半，否则按照fedavg
                    min_neighbor_id = min(close_neighbors, key=close_neighbors.get)
                    min_distance = calculate_weight(close_neighbors[min_neighbor_id],threshold=similarity_threshold)#选择小于阈值的最小距离的节点的倒数作为自身的权重
                    inverse_distances = { k : calculate_weight(v,threshold=similarity_threshold) for k, v in c.selected_weigh.items()}
                    total_inverse_weight = sum(inverse_distances.values()) + min_distance
                else:
                    min_distance = self.train_data_local_num_dict[c.client_idx]
                    total_inverse_weight = 0
                    total_inverse_weight += min_distance
                    for neigh_index in c.selected_neighbors:
                        total_inverse_weight += self.train_data_local_num_dict[neigh_index]
                    inverse_distances = { k : self.train_data_local_num_dict[k] for k in c.selected_neighbors}
                # 自身权重
                #W_run = self.train_data_local_num_dict[c.client_idx] / total_num
                W = min_distance / total_inverse_weight
                global_mean += all_running_means[c.client_idx] * W#_run #初始化聚合模型和global_mean
                for k in aggregate_weight[c.client_idx].keys():
                    aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]#初始化
                # 计算其他节点的权重并更新聚合权重和global_mean
                for neigh_index in c.selected_neighbors:
                    #w_run = self.train_data_local_num_dict[neigh_index] / total_num
                    w = inverse_distances[neigh_index] / total_inverse_weight
                    for k in aggregate_weight[c.client_idx].keys():
                        aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
                    global_mean += all_running_means[neigh_index] * w#_run
                    
                c.global_mean = global_mean
                
            for c in self.client_list:    
                self.w_per_globals[c.client_idx] = aggregate_weight[c.client_idx]   
            
            self.logger.info('Doing local aggregation!')
            
            
                 

    def _aggregate_func(self,all_running_means,begin = 1):#FedAvg平均聚合
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
                for neigh_index in c.selected_neighbors:
                    total_num += self.train_data_local_num_dict[neigh_index]
                W = self.train_data_local_num_dict[c.client_idx] / total_num
                global_mean += all_running_means[c.client_idx] * W #初始化聚合模型和global_mean
                for k in aggregate_weight[c.client_idx].keys():
                    aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]#初始化
                for neigh_index in c.selected_neighbors:
                    w = self.train_data_local_num_dict[neigh_index] / total_num
                    for k in aggregate_weight[c.client_idx].keys():
                        aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
                    global_mean += all_running_means[neigh_index] * w
                c.global_mean = global_mean
            for c in self.client_list:    
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
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops+= [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops)/len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops
        
    

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
        
    def change_neighbors(self, grap=1, similarity_threshold = 2):#grap=1表示图是动态的，grap=0表示图是静态的需要自己改变拓扑邻居节点
        connections_to_remove = set()
        connections_to_add = set()
        for client in self.client_list:
            if client.signal == 1:#需要补充额外的二阶邻居节点
                need_num = client.max_num - len(client.selected_neighbors)
                if len(client.second_neighbors) < need_num:#如果二阶邻居的数量小于差值的话，全部补上
                    for second_id in client.second_neighbors.keys():
                        if self.client_list[second_id].signal == 1:
                            self.client_list[second_id].selected_weigh.update({client.client_idx : client.second_neighbors[second_id]})
                            self.client_list[second_id].selected_neighbors.add(client.client_idx)
                            client.selected_weigh.update({second_id : client.second_neighbors[second_id]})
                            client.selected_neighbors.add(second_id)
                            if grap == 0:#静态图还需要添加拓扑连接
                                add_pair = frozenset([client.client_idx, second_id])
                                connections_to_add.add(add_pair)
                                #self.add_connection(self.graph,client.id,second_id)
                                client.neigh_distance.update({second_id:client.second_neighbors[second_id]})
                                self.client_list[second_id].neigh_distance.update({client.client_idx: client.second_neighbors[second_id]})
                            if len(self.client_list[second_id].selected_neighbors) == self.client_list[second_id].max_num:
                                self.client_list[second_id].signal = 0#饱和，之后不需要再添加额外的邻居节点
                elif len(client.second_neighbors) >= need_num:#如果二阶邻居的数量大于差值的话，择优补上
                    sorted_second_neighbors = dict(
                        sorted(client.second_neighbors.items(), key=lambda item: item[1])
                    )
                    for second_id in sorted_second_neighbors.keys(): 
                        if self.client_list[second_id].signal == 1:
                            self.client_list[second_id].selected_weigh.update({client.client_idx : client.second_neighbors[second_id]})
                            self.client_list[second_id].selected_neighbors.add(client.client_idx)
                            client.selected_weigh.update({second_id : client.second_neighbors[second_id]})
                            client.selected_neighbors.add(second_id)
                            if grap == 0:#静态图还需要获取拓扑连接
                                #self.add_connection(self.graph,client.id,second_id)
                                add_pair = frozenset([client.client_idx, second_id])
                                connections_to_add.add(add_pair)
                                client.neigh_distance.update({second_id:sorted_second_neighbors[second_id]})
                                self.client_list[second_id].neigh_distance.update({client.client_idx:sorted_second_neighbors[second_id]})
                            if len(self.client_list[second_id].selected_neighbors) == self.client_list[second_id].max_num:
                                self.client_list[second_id].signal = 0#饱和，不需要再添加额外的邻居节点
                            if len(client.selected_neighbors) == client.max_num:#本地也饱和
                                client.signal = 0
                                break#client已经达到了通信瓶颈，现在可以跳到下一个client
        if grap == 0:#必须跳出循环才能进行删除和添加
            for pair in connections_to_add:
                id_list = list(pair)
                id1, id2 = sorted(id_list)
                self.add_connection(self.graph,id1,id2)                
        
        for client in self.client_list:#尽量选择高质量的邻居节点，如果没有那也只能将就着聚合
            '''
            print("-----------------------最终------------------")
            print(f"client_id:{client.id}")
            print(f"client.neigh_distance:{client.neigh_distance}")
            print(f"client.selected_neighbors:{client.selected_neighbors}")
            print(f"client.signal:{client.signal}")
            '''
            if client.signal == 1:
                sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
                for neigh_id in sorted_neigh_distance.keys():
                    if neigh_id not in client.selected_neighbors:
                        if self.client_list[neigh_id].signal == 1:
                            self.client_list[neigh_id].selected_weigh.update({client.client_idx : sorted_neigh_distance[neigh_id]})
                            client.selected_weigh.update({neigh_id : sorted_neigh_distance[neigh_id]})
                            self.client_list[neigh_id].selected_neighbors.add(client.client_idx)
                            client.selected_neighbors.add(neigh_id)
                            if len(self.client_list[neigh_id].selected_neighbors) == self.client_list[neigh_id].max_num:
                                self.client_list[neigh_id].signal = 0#饱和，不需要再添加额外的邻居节点
                        if len(client.selected_neighbors) == client.max_num:
                            client.signal = 0#最后保证所有的客户端都是满员的
                            break
        
        if grap == 0:#静态图还需要删除拓扑连接
            for client in self.client_list:
                if client.signal == 0:#不需要补充额外的二阶邻居节点，检查有没有需要删除的节点
                    for neigh_id in self.graph.adj[client.client_idx]:
                        if neigh_id not in client.selected_neighbors and client.client_idx not in self.client_list[neigh_id].selected_neighbors:#两边都看不上对方
                            if self.client_list[neigh_id].signal == 0:
                            #self.remove_connection(self.graph,client.id,neigh_id)
                                remove_pair = frozenset([client.client_idx, neigh_id])
                                connections_to_remove.add(remove_pair)
            for pair in connections_to_remove:
                id_list = list(pair)
                id1, id2 = sorted(id_list)
                self.remove_connection(self.graph,id1,id2)
                    
                            
                        #self.selected_clients_dict[neigh1_id].selected_neighbors.append(second_id)
        
        
    def select_neighbors(self,client, similarity_threshold = 2):
        selected_neighbors = set()
        selected_weigh = {}
        client_mean = client.running_mean.cpu().detach().numpy().reshape(1, -1)
        client1_mean_prob = F.softmax(torch.tensor(client_mean), dim=1).numpy()
        for neigh_id in self.graph.adj[client.client_idx]:
            other_client_mean = self.client_list[neigh_id].running_mean.cpu().detach().numpy().reshape(1, -1)
        # 计算余弦相似度
            client2_mean_prob = F.softmax(torch.tensor(other_client_mean), dim=1).numpy()
            distance = np.sum(client1_mean_prob * np.log(client1_mean_prob / client2_mean_prob))* 1000
            client.neigh_distance.update({neigh_id : distance})#统计直接相连的邻居的距离
            
        sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
        for neigh_id in sorted_neigh_distance:
            if sorted_neigh_distance[neigh_id] < similarity_threshold:
                selected_neighbors.add(neigh_id)
                selected_weigh.update({neigh_id : client.neigh_distance[neigh_id]})
            else:#以大于阈值的最小的kl散度距离作为衡量是否二跳邻居节点的依据
                if client.threshold > sorted_neigh_distance[neigh_id]:
                    client.threshold = sorted_neigh_distance[neigh_id]
            if len(selected_neighbors) == client.max_num:
                break
            
        client.selected_neighbors = selected_neighbors
        client.selected_weigh = selected_weigh
        if len(selected_neighbors) < client.max_num:
            client.signal = 1#需要额外的二阶节点

    
    def select_neighbors_two(self,client):
        needed_neighbors = []
        for neigh_id in self.graph.adj[client.client_idx]:
            if self.client_list[neigh_id].signal == 1:
                needed_neighbors.append(neigh_id)#统计需要额外邻居节点信息的客户端id号
        if len(needed_neighbors) > 0:
            for neigh1_id in needed_neighbors:
                client1_mean = self.client_list[neigh1_id].running_mean.cpu().detach().numpy().reshape(1, -1)
                client1_mean_prob = F.softmax(torch.tensor(client1_mean), dim=1).numpy()
                for neigh2_id in self.graph.adj[client.client_idx]:
                    if neigh1_id != neigh2_id and neigh2_id not in self.graph.adj[neigh1_id]:#二阶邻居不与自身相等，二阶邻居不在自己的邻居中
                        client2_mean = self.client_list[neigh2_id].running_mean.cpu().detach().numpy().reshape(1, -1)
                        client2_mean_prob = F.softmax(torch.tensor(client2_mean), dim=1).numpy()
                        similarity1 = np.sum(client1_mean_prob * np.log(client1_mean_prob / client2_mean_prob))* 1000
                        similarity2 = np.sum(client2_mean_prob * np.log(client2_mean_prob / client1_mean_prob))* 1000
                        #print(f"second similarity:{similarity}")
                        if similarity1 <= self.client_list[neigh1_id].threshold and similarity2 <= self.client_list[neigh2_id].threshold:
                            self.client_list[neigh1_id].second_neighbors.update({neigh2_id : similarity1})#每个client汇总收到的二阶邻居信息
                            self.client_list[neigh2_id].second_neighbors.update({neigh1_id : similarity2})
                            
def calculate_weight(distance,threshold = 8,A = 1,mu = 0):
    sigma = -0.0555 + 0.8825 * threshold - 0.0008 * threshold**2
    if distance <= threshold:
        return A * np.exp(-(distance**2) / (2 * sigma**2))
    else:
        gauss_weight_at_threshold = A * np.exp(-(distance**2) / (2 * sigma**2)) * np.exp(-0.8*(distance-threshold))
        return gauss_weight_at_threshold
        



