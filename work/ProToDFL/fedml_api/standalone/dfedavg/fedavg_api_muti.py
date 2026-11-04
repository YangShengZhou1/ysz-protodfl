import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from fedml_api.standalone.dfedavg.clients import Client
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon, cosine
from torch import nn


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
        self.graph = self.generate_graph()
        self.sub = 0
        self.add = 0
        w_global_cifar10 = self.model_trainers[0].get_trainable_params(device)
        w_global_cifar100 = self.model_trainers[1].get_trainable_params(device)
        self.w_per_globals = [copy.deepcopy(w_global_cifar10) for _ in range(self.args.client_num_in_total // 2)] + \
                    [copy.deepcopy(w_global_cifar100) for _ in range(self.args.client_num_in_total // 2)]
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights,  tst_results = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, 0)
            self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
            
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

        
    def generate_graph(self):
        # time-varying topology
        while True:
            graph = nx.random_geometric_graph(self.args.client_num_in_total,0.25)
            if nx.is_k_edge_connected(graph, 1)==True:
                break
        return graph

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        params_cifar10 = self.model_trainers[0].get_trainable_params(self.device)
        params_cifar100 = self.model_trainers[1].get_trainable_params(self.device)
        half_clients = self.args.client_num_in_total // 2
        for client_idx in range(self.args.client_num_in_total):
            if client_idx < half_clients:
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.args, self.device, model_trainer[0], self.logger,params_cifar10)
            else:
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.args, self.device, model_trainer[1], self.logger,params_cifar100)
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
            weights,_ = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, 0)
            self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
            #num_comm_params[c.client_idx] = num_comm_params
            #c.neighbor_dict = self._benefit_choose(c.client_idx, self.args.client_num_in_total,
            #                                      self.args.client_num_per_round, self.args.cs)
        #self.graph = self.generate_graph()
        # device = {device} cuda:0apply mask to init weights
        #threshold = 6#逐步缩小阈值 cifar10,resnet18
        threshold = [6,86]#逐步缩小阈值 cifar100,resnet18
        #threshold = 120#逐步缩小阈值 tiny,resnet18
        #threshold = 6#逐步缩小阈值 cifar10,vgg11
        #threshold = 65#逐步缩小阈值 cifar100,vgg11
        ##########################################################################################################
        ###################################****协议全局训练开始****#################################################
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
            self.logger.info("################Communication round : {}".format(round_idx))
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            w_locals = []
            self.logger.info('add number:{}'.format(self.add))
            self.logger.info('sub number:{}'.format(self.sub))
            for client in self.client_list:#初始化
                client.signal = 0
                client.num = 0
                client.selected_weigh.clear()#选中节点的余弦距离
                client.selected_neighbors.clear()#选中节点集合
                client.second_neighbors.clear()#二阶邻居节点的余弦距离
                client.neigh_distance.clear()#一阶邻居节点的余弦距离
                client.threshold = 500
            
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
    
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, str(c.client_idx)))#该客户端在本轮训练中活跃，记录训练日志
                
                #客户端c选择邻居客户端节点
                self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
            for client in self.client_list:
                if client.wait_signal != 1:
                    self.select_neighbors(client,similarity_threshold = threshold)
                    self.logger.info("-----------------------最初------------------")
                    self.logger.info('client_id: {}'.format(client.client_idx))
                    self.logger.info('client.neigh_distance: {}'.format(client.neigh_distance))
                else:
                    self.compare_exper(client)
            for client in self.client_list:
                self.select_neighbors_two(client)
                
            self.change_neighbors(grap=0,rounds = round_idx,similarity_threshold = threshold)#cifar10 2.15 cifar100 1.2 fmnist 1.6
            self._aggregate_func()#聚合模型
            for c in self.client_list:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))
                # update dataset
                # update meta components in personal network
                weights, tst_results = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, round_idx)
                self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
                #num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)

            self._test_on_all_clients(round_idx)
            #if threshold >= 40 : #cifar100,vgg11
            #    threshold = max(threshold- 1 * round_idx , 40)
            #else:
            #    threshold = max(threshold- 0.2 * round_idx , 20)
            #if threshold >= 5 : #cifar10,vgg11
            #    threshold = max(threshold- 0.1 * round_idx , 4)
            #else:
            #    threshold = max(threshold- 0.01 * round_idx , 2.5)
            #if threshold >= 50 : #tiny,resnet18
            #    threshold = max(threshold- 4 * round_idx , 40)
            #else:
            #    threshold = max(threshold- 0.2 * round_idx , 26)
            if threshold[0] >= 5 : #cifar10,resnet18
                threshold[0] = max(threshold[0]- 0.05 * round_idx , 5)
            else:
                threshold[0] = max(threshold[0]- 0.01 * round_idx , 4)
            if threshold[1] >= 50 : #cifar100,resnet18
                threshold[1] = max(threshold[1]- 2 * round_idx , 50)
            else:
                threshold[1] = max(threshold[1]- 0.5 * round_idx , 22)
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
    
    
    def _aggregate_func(self):  # FedAvg平均聚合
      aggregate_weight = {}
      for c in self.client_list:
          aggregate_weight.update({c.client_idx: {k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}})
          
      for c in self.client_list:
          total_num = self.train_data_local_num_dict[c.client_idx]
          for neigh_index in c.selected_neighbors:
              total_num += self.train_data_local_num_dict[neigh_index]
          
          W = self.train_data_local_num_dict[c.client_idx] / total_num
          
          for k in aggregate_weight[c.client_idx].keys():
              if not self._is_classifier_layer(k):
                  aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]
          
          for neigh_index in c.selected_neighbors:
              w = self.train_data_local_num_dict[neigh_index] / total_num
              for k in aggregate_weight[c.client_idx].keys():
                  if not self._is_classifier_layer(k):
                        aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
          
      for c in self.client_list:
        for k in aggregate_weight[c.client_idx].keys():
            if not self._is_classifier_layer(k):
                self.w_per_globals[c.client_idx][k] = aggregate_weight[c.client_idx][k]
        
    def _is_classifier_layer(self, layer_name):
        # 识别分类器层的简单实现，根据实际模型层名称进行调整
        classifier_keywords = ['head', 'linear']
        for keyword in classifier_keywords:
            if keyword in layer_name:
                return True
        return False
    
    def change_neighbors(self, grap=1, rounds = 0, similarity_threshold):#grap=1表示图是动态的，grap=0表示图是静态的需要自己改变拓扑邻居节点
        connections_to_remove = set()
        connections_to_add = set()
        for client in self.client_list:
            if rounds >= 100:
                if len(client.second_neighbors) == 0:#没有需要补充的二阶邻居节点，停止邻居选择
                    if client.wait_signal == 0:#在本轮中无邻居优化，下一轮跳过
                        client.wait_signal = 2
                        client.wait_time[0] = 0
                        client.wait_time[1] = 0
                    elif client.wait_signal == 1:#在本轮中跳过邻居
                        client.wait_time[0] -= 1
                        if client.wait_time[0] == 0:#最后一轮停止等待
                            client.wait_signal = 2#设置为过渡状态,再观察在下一轮中是否有邻居节点更新
                    elif client.wait_signal == 2 :#还是没有邻居节点更新，延迟等待时间
                        if 2**client.wait_time[1] >16:
                            client.wait_time[0] = min(100,client.wait_time[1]*2)
                            client.wait_time[1] = min(100,client.wait_time[1]*2)
                        else:
                            client.wait_time[0] = 2**client.wait_time[1]
                            client.wait_time[1] = 2**client.wait_time[1]
                        client.wait_signal = 1
                        continue#跳过这个邻居
                elif client.wait_signal == 2:#出现了新的连接节点更新，等待时间重置
                    client.wait_time[0] = 0
                    client.wait_time[1] = 0
                    client.wait_signal = 0
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
        
        
    def select_neighbors(self, client, similarity_threshold):
        selected_neighbors = set()
        selected_weigh = {}
        if client.client_idx < 50 :
            threshold = similarity_threshold[0]
        else :
            threshold = similarity_threshold[1]
        client_mean = client.running_mean.to(self.device).detach().reshape(1, -1)
        client1_mean_prob = F.softmax(client_mean, dim=1)
        for neigh_id in self.graph.adj[client.client_idx]:
            if self.client_list[neigh_id].wait_signal == 1:
                selected_neighbors.add(neigh_id)#直接默认邻居，不能改变拓扑结构
                continue
            other_client_mean = self.client_list[neigh_id].running_mean.to(self.device).detach().reshape(1, -1)
        # 计算余弦相似度
            client2_mean_prob = F.softmax(other_client_mean, dim=1)
            distance = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
            distance = distance.item()
            client.neigh_distance.update({neigh_id : distance})#统计直接相连的邻居的距离
            
        sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
        for neigh_id in sorted_neigh_distance:
            if sorted_neigh_distance[neigh_id] < threshold:
                selected_neighbors.add(neigh_id)
                selected_weigh.update({neigh_id : client.neigh_distance[neigh_id]})
                client.num += 1
            else:#以大于阈值的最小的kl散度距离作为衡量是否二跳邻居节点的依据
                if client.threshold > sorted_neigh_distance[neigh_id]:
                    client.threshold = sorted_neigh_distance[neigh_id]
            if len(selected_neighbors) == client.max_num:#不再需要额外的二阶邻居了
                client.threshold = client.neigh_distance[neigh_id]
                break
        if client.threshold ==  500:
           client.threshold = threshold   
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
                client1_mean = self.client_list[neigh1_id].running_mean.to(self.device).detach().reshape(1, -1)
                client1_mean_prob = F.softmax(client1_mean, dim=1)
                for neigh2_id in self.graph.adj[client.client_idx]:
                    if neigh1_id != neigh2_id and neigh2_id not in self.graph.adj[neigh1_id]:#二阶邻居不与自身相等，二阶邻居不在自己的邻居中
                        client2_mean = self.client_list[neigh2_id].running_mean.to(self.device).detach().reshape(1, -1)
                        client2_mean_prob = F.softmax(client2_mean, dim=1)
                        similarity1 = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob))* 1000
                        similarity2 = torch.sum(client2_mean_prob * torch.log(client2_mean_prob / client1_mean_prob))* 1000
                        similarity1 = similarity1.item()  # 转换为Python标量
                        similarity2 = similarity2.item()  # 转换为Python标量
                        #print(f"second similarity:{similarity}")
                        if similarity1 <= self.client_list[neigh1_id].threshold and similarity2 <= self.client_list[neigh2_id].threshold:
                            self.client_list[neigh1_id].second_neighbors.update({neigh2_id : similarity1})#每个client汇总收到的二阶邻居信息
                            self.client_list[neigh2_id].second_neighbors.update({neigh1_id : similarity2})

