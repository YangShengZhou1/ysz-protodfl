import copy
import logging
import pickle
import random
import torch
import networkx as nx
from tqdm import tqdm
import numpy as np


from fedml_api.standalone.ditto.client import Client


class DittoAPI(object):
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
        self.graph = self.generate_graph()
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        params_cifar10 = self.model_trainers[0].get_trainable_params(self.device)
        params_cifar100 = self.model_trainers[1].get_trainable_params(self.device)
        self.w_per_globals = [{k: v.to(self.device).detach().clone() for k, v in params_cifar10.items()} for _ in range(self.args.client_num_in_total // 2)] + \
                             [{k: v.to(self.device).detach().clone() for k, v in params_cifar100.items()} for _ in range(self.args.client_num_in_total // 2)]
        

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        half_clients = self.args.client_num_in_total // 2
        for client_idx in range(self.args.client_num_in_total):
            if client_idx < half_clients:
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.args, self.device, model_trainer[0], self.logger)
            else:
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.args, self.device, model_trainer[1], self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")
        
    def generate_graph(self):
        # time-varying topology
        while True:
            graph = nx.random_geometric_graph(self.args.client_num_in_total,0.25)
            if nx.is_k_edge_connected(graph, 1)==True:
                break
        return graph
    
    def compare_exper(self,client):
        selected_neighbors = set()
        for neigh_id in self.graph.adj[client.client_idx]:
            selected_neighbors.add(neigh_id)
        client.selected_neighbors = selected_neighbors
        
    def _to_device(self, state_dict, device):
        return {k: v.to(device) for k, v in state_dict.items()}

    def train(self):
        params_cifar10 = self.model_trainers[0].get_trainable_params(self.device)
        params_cifar100 = self.model_trainers[1].get_trainable_params(self.device)
        w_pers = [{k: v.to(self.device).detach().clone() for k, v in params_cifar10.items()} for _ in range(self.args.client_num_in_total // 2)] + \
                             [{k: v.to(self.device).detach().clone() for k, v in params_cifar100.items()} for _ in range(self.args.client_num_in_total // 2)]
        #client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
        #                                           self.args.client_num_per_round)
        for c in self.client_list:
            self.compare_exper(c)
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
            self.logger.info("################Communication round : {}".format(round_idx))
            #w_locals = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            
            self.logger.info("client_indexes = " + str(c.client_idx))
            self._aggregate_func()

            for c in self.client_list:
                # update dataset

                
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))
                # update meta components in personal network
                weights,num_comm_params = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, round_idx)
                
                w_pers[c.client_idx] = c.local_train(w_pers[c.client_idx], {k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()},round_idx)
                self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
                #w_locals.append((client.get_sample_number(), copy.deepcopy(weights)))
                #self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update global weights
            

            # test results at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(w_pers, round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(w_pers, round_idx)
        #self.record_avg_inference_flops(w_global)

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

    def _local_test_on_all_clients(self, w_pers, round_idx):

        self.logger.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            test_local_metrics = client.local_test(w_pers[client_idx], True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        test_acc = sum([np.array(test_metrics['num_correct'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        self.stat_info["test_acc"].append(test_acc)
        self.logger.info(stats)

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
        self.stat_info["test_acc"] = []
        self.stat_info["final_masks"] = []
        
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
        classifier_keywords = ['linear', 'fc', 'classifier']
        for keyword in classifier_keywords:
            if keyword in layer_name:
                return True
        return False

