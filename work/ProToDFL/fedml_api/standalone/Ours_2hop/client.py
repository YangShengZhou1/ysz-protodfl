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

from fedml_api.standalone.Ours_2hop import client
from fedml_api.standalone.Ours_2hop.client import Client


class global_API(object):
    def __init__(self, dataset, device, args, model_trainer, logger):# model_trainer即管理稀疏神经网络的类，其中包含对稀疏神经网络的各项操作
        self.logger = logger
        self.device = args.device
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



    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        #self.generate_connection_matrix()# 保存dist矩阵，存储每个client和其他client的距离
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        # for first initialization
        w_global = self.model_trainer.get_model_params()
        #params = self.model_trainer.get_trainable_params()
        model_weight = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        
        num_comm_params = {}
        # 初始化
        ##########################################################################################################
        ###################################模型初始化：w_global和mask##############################################
        ##########################################################################################################
        #for clnt in range(self.args.client_num_in_total):
        #    w_per_mdls.append(copy.deepcopy(w_global))#本地客户端将本地模型初始化为全局模型w_global
        #   updates_matrix.append(copy.deepcopy(w_global))
        #   for name in mask_pers_local[clnt]:
        #       w_per_mdls[clnt][name] = w_global[name] #model=w_global
        #       updates_matrix[clnt][name] = updates_matrix[clnt][name] - updates_matrix[clnt][name]#初始化 updates_matrix 中的元素，使其都等于零
               
        for c in self.client_list:
            #c.model_trainer.set_model_params(model_weight[c.client_idx])
            weights, update, num_comm_param, tst_results = c.train(model_weight[c.client_idx], 0)
            model_weight[c.client_idx] = weights
            num_comm_params[c.client_idx] = num_comm_param
            self._neigh_choose(0, c, self.args.client_num_in_total,
                                                  self.args.client_num_per_round, self.args.cs)
        
        #在全局初始化阶段，每个本地客户端先进行K轮本地训练获得本地模型的更新梯度、以及loss损失
        #print("num_comm_params:")
        #print(len(num_comm_params))
        #print(num_comm_params)

        
        ##########################################################################################################
        ###################################Dis-pFL协议全局训练开始#################################################
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round), desc="Training Progress"):
            self.logger.info("################Communication round : {}".format(round_idx))
            #active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])#根据概率分布随机选择哪些客户端在当前通信轮次中活跃（1活跃，0表示不活跃）
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])
            #对每个客户端进行操作
            for c in self.client_list:
                c.active_ths_rnd = active_ths_rnd[c.client_idx]
                c.max_num = len(c.neigh_one_hop_choose)
            #for c in self.client_list:
                if self.args.choose == 2 :#原版方案
                    if active_ths_rnd[c.client_idx] == 1:
                        c.caculate_nei_priority(self.client_list,gama=0.7)#选择1-hop节点
                        c.send_2_hop_neigh(self.client_list)#发送2-hop节点
                elif self.args.choose == 1 :#只选择1-hop节点
                    if active_ths_rnd[c.client_idx] == 1:
                        c.caculate_nei_priority(self.client_list,gama=1)
                elif self.args.choose == 0 :#random choose
                    if active_ths_rnd[c.client_idx] == 1:
                        c.caculate_nei_random(self.client_list)
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.neighbor_dict))#该客户端在本轮训练中活跃，记录训练日志
                if self.args.choose == 2 :#原版方案
                    if active_ths_rnd[c.client_idx] == 1:
                        c.neigh_2_hop_choose(self.client_list)#选择2-hop节点
                # 更新dist_locals 矩阵
                #for tmp_idx in nei_indexs:
                #    if tmp_idx != clnt_idx:
                #        dist_locals[clnt_idx][tmp_idx],_ = hamming_distance(mask_pers_local[clnt_idx], mask_pers_shared_lstrd[tmp_idx])

                #客户端c选择邻居客户端节点
                
                if self.args.cs!="full":
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(c.neighbor_dict), self.args.cs))
                else:
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(c.neighbor_dict), self.args.cs))
                #Update each client's local model and the so-called consensus model
                #if active_ths_rnd[c.client_idx] == 1:
                if self.args.aggregate == 3:
                    if active_ths_rnd[c.client_idx] == 1:
                        self._aggregate_func(c, self.client_list,round_idx,model_weight)#聚合模型
                
                if self.args.aggregate == 0:
                    if active_ths_rnd[c.client_idx] == 1:
                        self._aggregate_avg(c, self.client_list,round_idx,model_weight)
                
            for c in self.client_list:# 聚合好模型后，设置client进行local training
                w_local = model_weight[c.client_idx]
                #weights, update, num_comm_param, tst_results = c.train(copy.deepcopy(w_local), round_idx)
                #num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(w_local, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                #tst_results_ths_round.append(tst_results)
            self._local_test_on_all_clients_new_mask(final_tst_results_ths_round, round_idx)
            for c in self.client_list:# 聚合好模型后，设置client进行local training
                #w_local = c.model_trainer.get_model_params()
                #self.logger.info("client_model{} params = {}".format(c.client_idx, model_weight[c.client_idx]))
                weights, update, num_comm_param, tst_results = c.train(model_weight[c.client_idx], round_idx)
                model_weight[c.client_idx] = weights
                #self.logger.info("client_model{} params = {}".format(c.client_idx, weights))
                #self.logger.info("client_model{} params = {}".format(c.client_idx, weights))
                num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)
            #所有客户端完成模型和参数的聚合以及更新后
            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
            for c in self.client_list:
                c.neigh_one_hop_choose.clear()
                c.neigh_two_hop_choose.clear()
                c.neighbor_dict.clear()
                self._neigh_choose(0, c, self.args.client_num_in_total,
                                                  self.args.client_num_per_round, self.args.cs)
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

    def _neigh_choose(self, round_idx, c, client_num_in_total, client_num_per_round, cs = False):
        #根据不同的通信拓扑结构选择邻居客户端
        if client_num_in_total == client_num_per_round:
            # 表示每个客户端都可以与所有其他客户端通信且没有带宽限制，那么就返回所有客户端的索引
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            #c.add_or_update_neighbor(client_indexes,1)
            for idx in client_indexes:
                c.add_or_update_neighbor(idx, 1)

        if cs == "random":
            # 则随机选择可用客户端，数量不超过 client_num_per_round，并确保不包括当前客户端。
            num_clients = min(client_num_per_round, client_num_in_total)
             # 随机选择客户端索引
            client_indexes = np.random.choice([i for i in range(client_num_in_total) if i != c.client_idx],
                                      num_clients, replace=False)
            for idx in client_indexes:
                c.add_or_update_neighbor(idx, 1)

        elif cs == "ring":
            # 使用环形拓扑结构，选择当前客户端的左侧和右侧客户端作为邻居。
            left = (c.client_idx - 1 + client_num_in_total) % client_num_in_total
            right = (c.client_idx + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])
            #c.add_or_update_neighbor(client_indexes,1)
            for idx in client_indexes:
                c.add_or_update_neighbor(idx, 1)

        elif cs == "full":
            # 用全连接拓扑结构，选择在本轮活跃的客户端作为邻居，并确保不包括当前客户端。
            client_indexes = np.array(np.where(True)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes==c.client_idx)[0]))
            #c.add_or_update_neighbor(client_indexes,1)
            for idx in client_indexes:
                c.add_or_update_neighbor(idx, 1)
        # 将选中的客户端索引存储到c.nei_index
    
    def _aggregate_avg(self, client,client_list,round):#aggregate like FedAvg
        self.logger.info('Doing local aggregation!')
        weights = client.model_trainer.get_model_params()
        model = copy.deepcopy(client.model_trainer.model)
        model.load_state_dict(weights)
        Sum_num = client.local_sample_number
        accumulated_grad = {k: torch.zeros_like(v) for k, v in client.grad_accumulator.items()}
        model_weight = client.last_model
        for index in client.neighbor_dict:
            Sum_num += client_list[index].local_sample_number
        
        for index in client.neighbor_dict:
            #if isinstance(client.neighbor_dict[index], dict) :
                neighbor_grad = client_list[index].grad_accumulator
                weight = client_list[index].local_sample_number/Sum_num
                for k in accumulated_grad:
                    accumulated_grad[k] += weight * neighbor_grad[k]
        model_weight = client.last_model
        for k in weights:
            model_weight[k] -= self.lr* (self.lr_decay**round)*((client.local_sample_number/Sum_num) * client.grad_accumulator[k] + accumulated_grad[k])

        client.model_trainer.set_model_params(model_weight)#更新聚合模型

    def _aggregate_func(self,client,client_list,round,model_weights,alpha=0.5):
        beta = self.args.beta
        self.logger.info('Doing local aggregation!')
        weights = model_weights[client.client_idx]
        #client.model_trainer.set_model_params(weights)
        #model = copy.deepcopy(client.model_trainer.model)
        client.model_trainer.set_model_params(weights)
        client.log_probs = client.validate_model(client.model_trainer.model)
        # 使用接收到的梯度计算输出
        neighbor_info = {}
        for index in client.neighbor_dict:
           # 获取当前邻居的模型梯度
           #print(f"Type of neighbor_dict[index]: {type(client.neighbor_dict.get(index))}")  # Debugging print
           #print(f"Value of neighbor_dict[index]: {client.neighbor_dict.get(index)}")  # Debugging print
           #if isinstance(client.neighbor_dict.get(index), dict) and client.neighbor_dict[index]['selected']:
           #print(client.neighbor_dict[index])
           if isinstance(client.neighbor_dict[index], dict) :
                neighbor_gradients = client_list[index].grad_accumulator
                # 使用邻居的梯度更新模型
                updated_model = client.apply_gradients(neighbor_gradients,round)
                # 将模型参数加载到模型实例
                #model2 = copy.deepcopy(client.model_trainer.model)
                client.model_trainer.model.load_state_dict(updated_model)
                # 在本地数据集上验证更新后的模型
                log_probs = client.validate_model(client.model_trainer.model)
                dist = client.calculate_distance(client.log_probs,log_probs)
                #neighbor_info[index] = {'dist': dist}#字典存储蒸馏距离
                cos = client.cosine_distance_accumulators(client.grad_accumulator,neighbor_gradients)
                neighbor_info[index] = {'cos': cos,'dist': dist}#字典存储梯度余弦距离
                #self.logger.info("cos: {}, dist: {}".format(cos, dist))
           else:
                #print(client.neighbor_dict[index])
                neighbor_info[index] = {'dist': 0, 'cos': 0}
        accumulated_grad = {k: torch.zeros_like(v) for k, v in client.grad_accumulator.items()}
        # 计算总距离
        total_dist = sum(info['dist'] for info in neighbor_info.values())
        total_cos = sum(info['cos'] for info in neighbor_info.values())
        total_weight = 0
        for index in client.neighbor_dict:
            if client.neighbor_dict[index]['selected'] == 1:
                dist_weight = alpha * (neighbor_info[index]['dist'] / total_dist) if total_dist > 0 else (1 / len(neighbor_info))
                cos_weight = (1 - alpha) * (neighbor_info[index]['cos'] / total_cos) if total_cos > 0 else (1 / len(neighbor_info))
                total_weight += max(cos_weight - dist_weight, 0)
                #client.neighbor_dict[index]['weight'] = dist_weight + cos_weight
                
            else:
                client.neighbor_dict[index]['weight'] = 0#未被选择，不参与模型聚合
        if total_weight != 0:
            for index in client.neighbor_dict:
                if client.neighbor_dict[index]['selected'] == 1:
                    dist_weight = alpha * (neighbor_info[index]['dist'] / total_dist) if total_dist > 0 else (1 / len(neighbor_info))
                    cos_weight = (1 - alpha) * (neighbor_info[index]['cos'] / total_cos) if total_cos > 0 else (1 / len(neighbor_info))
                    weight = max(cos_weight-dist_weight,0) / total_weight
                    client.neighbor_dict[index]['weight'] = weight
                    neighbor_grad = client_list[index].grad_accumulator
                    for k in accumulated_grad:
                        accumulated_grad[k] += weight * neighbor_grad[k]
                

        

        # Combine the client's own gradient with the accumulated neighbor gradients
        model_weight = client.last_model
        for k in weights:
            model_weight[k] -= (beta * client.grad_accumulator[k] + (1 - beta) * accumulated_grad[k])
        model_weights[client.client_idx] = model_weight
            
        #更新聚合模型
        #model_weight = client.last_model_trainer.get_model_params()
        #updated_grad = sum(client.neighbor_dict[index]['weight']*client_list[index].grad_accumulator for index in client.neighbor_dict)
        #updated_weight = model_weight - (beta*client.grad_accumulator+(1-beta)*updated_grad)
        client.model_trainer.set_model_params(model_weight)#更新聚合模型
        
    

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
        
    def calculate_neighbor_cosine_similarities(self,client, client_list):
        """
        计算指定客户端的邻居节点之间的两两梯度余弦相似度。
ur
        :param client: 当前客户端。
        :param client_list: 包含所有客户端的列表。
        :return: 邻居节点之间的两两余弦相似度。
        """
        neighbor_indices = list(client.neighbor_dict.keys())
        cosine_scores = {}

        for i in range(len(neighbor_indices)):
            for j in range(i + 1, len(neighbor_indices)):
                idx1 = neighbor_indices[i]
                idx2 = neighbor_indices[j]
                grad1 = client_list[idx1].grad_accumulator
                grad2 = client_list[idx2].grad_accumulator
                cos_sim = client.cosine_distance_accumulators(grad1, grad2)
                cosine_scores[(idx1, idx2)] = cos_sim

        return cosine_scores


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
        
    def calculate_muti_neighbor(self, client, alpha=0.5):
        """
        对于每个客户端的邻居节点，计算它们之间的相似度，并更新邻居信息。

        :param client: 当前客户端。
        :param alpha: 相似度阈值。
        """
        # 计算邻居节点间的余弦相似度
        cosine_scores = self.calculate_neighbor_cosine_similarities(client, self.client_list)

        # 遍历计算出的相似度
        for (idx1, idx2), cos_sim in cosine_scores.items():
            if client.neighbor_dict[idx1]['selected'] == 1 and client.neighbor_dict[idx2]['selected'] == 1 :
                if cos_sim > alpha:
                    # 获取当前邻居的hop值
                    hop1 = client.neighbor_dict[idx1]['hop']
                    hop2 = client.neighbor_dict[idx2]['hop']

                    # 更新邻居信息，hop值增加1
                    self.client_list[idx1].add_or_update_neighbor(idx2, hop2 + 1, selected=0)
                    self.client_list[idx2].add_or_update_neighbor(idx1, hop1 + 1, selected=0)#这里先设置为false，是为了在下一轮计算邻居节点的时候进行公平计算



