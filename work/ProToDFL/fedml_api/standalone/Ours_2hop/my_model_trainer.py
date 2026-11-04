import copy
import logging
import math
import torch
from torch import nn
import numpy as np
import pdb
import torch

import torch.nn.functional as F

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.active_ths_rnd = 0
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.grad_accumulator = {}
        self.neighbor_dict = {}#include chosed 1_hop_nei and 2_hop_nei index
        self.neigh_one_hop_choose = {}#1_hop_nei
        self.neigh_two_hop_choose = {}#2_hop_nei
        self.loss = 0
        self.last_model = model_trainer.get_model_params()#记录上一轮global commun下的模型参数
        self.learning_rate=args.lr
        self.lr_decay = args.lr_decay
        self.device = args.device
        self.log_probs = []
        self.tao = 0#tao用来表示优先级筛选的阈值
        self.max_num = 8#最大通信的节点个数


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,w,round):
        # 计算通信参数的数量
        num_comm_params = self.model_trainer.count_communication_params(w)# 设置模型参数为传入的 w
        self.last_model = copy.deepcopy(w)
        self.model_trainer.set_model_params(w)#设置模型初始化参数w^0
        #self.last_model = copy.deepcopy(w)
        self.model_trainer.set_id(self.client_idx)# 在本地测试数据上进行测试，并记录测试结果（先测试训练前的模型效果）
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
        self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        # 在本地训练数据上进行训练
        self.loss = self.model_trainer.train(self.local_training_data, self.device, self.args, round)
        weights = self.model_trainer.get_model_params()# 获取训练后的模型参数
        #self.model_trainer.set_model_params(weights)# 将模型参数设置为训练后的weights
        tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)# 在本地测试数据上进行测试，并记录测试结果
        self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))
        self.grad_accumulator = {}# 计算权重更新
        for name in weights:
            self.grad_accumulator[name] =  self.last_model[name] - weights[name]
        self.logger.info("-----------------------------------")
        num_comm_params += self.model_trainer.count_communication_params(self.grad_accumulator)
        return weights, self.grad_accumulator, num_comm_params, tst_results

    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
    def add_or_update_neighbor(self, index, hop, selected=1, weight=0):
        """
        向字典中添加或更新一个:param neighbor_dict: 存储邻居节点信息的字典。
        :param index: 邻居节点的索引。
        :param hop: 到达该邻居节点的跳数。
        :param selected: 该邻居节点是否被选中参与通信。
        :param weight:该邻居节点参与更新模型的权重
        """
        if isinstance(index, np.ndarray):
            # 检查数组长度并获取单个元素
            if len(index) == 1:
                index = index[0]
            else:
                raise ValueError("Index array should have only one element")
        # Now, use index as a key in the dictionary
        if index not in self.neigh_one_hop_choose:
        # 如果索引不存在，创建新的条目
            if hop == 1:
                self.neigh_one_hop_choose[index] = {'hop': hop, 'selected': selected, 'weight' :weight}
            else:
                if index not in self.neigh_two_hop_choose:#也不在2-hop邻居索引字典当中
                    self.neigh_two_hop_choose[index] = {'hop': hop, 'selected': selected, 'weight' :weight}
            
#        else:
        # 如果索引已存在，更新信息
#            self.neigh_one_hop_choose[index]['hop'] = min(self.neighbor_dict[index]['hop'], hop)
        
        #self.neighbor_dict[index]['selected'] = selected

    
    def caculate_nei_priority(self, client_list , gama=0.7 ) :#根据计算权重筛选优先级节点,alpha,delta是超参数
        #print(self.neighbor_dict)
        #if round == 1:
        #    for index in self.neighbor_dict:
        #        self.neighbor_dict[index]['selected'] = 1
        #else:
           neighbor_priorities = {}
           #to_delete = []
           #min_loss = min(client.loss-self.loss for client in client_list)
           max_loss = max(abs(client_list[index].loss-self.loss) for index in self.neigh_one_hop_choose)
           for index in self.neigh_one_hop_choose:
                #if index < len(client_list):
                    # 归一化 num_comm_params
                    #norm_num_comm_params = num_comm_params[index] / max_num_comm_params if max_num_comm_params != 0 else 0

                    # 归一化 loss 差值
                    loss_diff = abs(client_list[index].loss - self.loss)
                    prior = loss_diff / max_loss if max_loss != 0 else 0

                    # 计算归一化的 prior
                    #alpha / (self.neighbor_dict[index]['hop']*norm_num_comm_params)  + delta * norm_loss_diff
                    prior = max(0, min(prior, 1))  # 确保 prior 在 0 到 1 之间
                    neighbor_priorities[index] = prior
                    
           #这里改为取neighbor_priorities的前gama%作为筛选依据
           cut = math.ceil(self.max_num * gama)  # 计算cut的值并向上取整

           # 获取前cut位的索引
           selected_indices = sorted(neighbor_priorities, key=neighbor_priorities.get, reverse=True)[:cut]
           # 如果需要获取第cut位的prior值，先确保cut不超出selected_indices的范围
           if cut <= len(selected_indices):
                self.tao = neighbor_priorities[selected_indices[cut-1]]  # 获取第cut位的prior值
           else:
                self.tao = 0  # 如果cut超出范围，则设置为0
           # 更新selected状态
           for index in self.neigh_one_hop_choose:
                if index in selected_indices:
                    self.neighbor_dict[index] = self.neigh_one_hop_choose[index]
                    
    def caculate_nei_random(self):#用作消融实验，当不引入邻居节点选择策略时
        if len(self.neigh_one_hop_choose) < self.max_num:
            for index in self.neigh_one_hop_choose:
                self.neighbor_dict[index] = self.neigh_one_hop_choose[index]
        else:
            selected_indices = self.neigh_one_hop_choose[:self.max_num]
            for index in self.neigh_one_hop_choose:
                if index in selected_indices:
                    self.neighbor_dict[index] = self.neigh_one_hop_choose[index]
                                      
        
                

    def send_2_hop_neigh(self, client_list):#在一跳邻居节点之间相互计算优先级，并为他们选择合适的二跳邻居
        for index1 in self.neighbor_dict:
           max_loss = max(abs(client_list[index].loss-client_list[index1].loss) for index in self.neighbor_dict)
           for index2 in self.neighbor_dict:
                if index1 != index2:
                    loss_diff = abs(client_list[index2].loss - self.loss)
                    prior = loss_diff / max_loss if max_loss != 0 else 0
                    if prior >= self.tao:
                        hop2 = self.neighbor_dict[index2]['hop']
                        client_list[index1].add_or_update_neighbor(index2, hop2 + 1, selected=1)#添加到1-hop或者2-hop邻居节点当中


    def apply_gradients(self, gradients,round):
        """
        Apply gradients to the model parameters
        :param gradients: A dictionary of gradients to apply.
        :return: Updated model parameters.
        """
        # Get current model parameters
        model_params = copy.deepcopy(self.last_model)

        # Ensure gradients is a dictionary
        if not isinstance(gradients, dict):
            raise TypeError("Expected gradients to be a dictionary")
        #model_params = model_params.to("cpu")
        # Update model parameters with gradients
        for param_key in model_params.keys():
            if param_key in gradients:
                # Ensure the gradient for this parameter is a tensor
                if not isinstance(gradients[param_key], torch.Tensor):
                    raise TypeError(f"Gradient for {param_key} is not a tensor")
                model_params[param_key].sub_(gradients[param_key])

        return model_params
    
    def neigh_2_hop_choose(self,client_list):
        delta_loss = 0
        neighbor_priorities = {}
        if len(self.neigh_two_hop_choose) > 0:
            if len(self.neigh_two_hop_choose) < (self.max_num-len(self.neighbor_dict)):
                for index in self.neigh_two_hop_choose:
                    self.neighbor_dict[index] = self.neigh_two_hop_choose[index]
            else:
                for index in self.neigh_two_hop_choose:
                    delta_loss = max(abs(client_list[index].loss-self.loss),delta_loss)
                for index in self.neigh_two_hop_choose:
                    loss_diff = abs(client_list[index].loss - self.loss)
                    prior = loss_diff / delta_loss if delta_loss != 0 else 0
                    # 计算归一化的 prior
                    #alpha / (self.neighbor_dict[index]['hop']*norm_num_comm_params)  + delta * norm_loss_diff
                    prior = max(0, min(prior, 1))  # 确保 prior 在 0 到 1 之间
                    neighbor_priorities[index] = prior
                cut = self.max_num-len(self.neighbor_dict)  # 计算cut的值
                # 获取前cut位的索引
                selected_indices = sorted(neighbor_priorities, key=neighbor_priorities.get, reverse=True)[:cut]
                # 如果需要获取第cut位的prior值，先确保cut不超出selected_indices的范围
                # 更新selected状态
                for index in self.neigh_two_hop_choose:
                    if index in selected_indices:
                        self.neighbor_dict[index] = self.neigh_two_hop_choose[index]
                    #else:
                    #    del self.neigh_two_hop_choose[index]
    
    def validate_model(self, model):
        """
        :param model: 要验证的模型。
        :param train_data: 本地训练数据集。
        :param device: 计算设备（例如 'cuda' 或 'cpu'）。
        :param criterion: 损失函数。
        :param args: 训练参数。
        :return: 验证结果，例如准确率或其他性能指标。
        """
        model.eval()  # 将模型设置为评估模式
        #criterion = nn.CrossEntropyLoss().to(self.device)
        all_log_probs = []
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x = x.to(self.device)
                logits = model(x)
                log_probs = F.log_softmax(logits, dim=1)  # 将logits转换为log-probabilities
                all_log_probs.append(log_probs)
        return all_log_probs

    
    def calculate_distance(self,log_probs1, log_probs2):
        """
        计算两个模型蒸馏输出之间的距离。
        """
        log_probs2 = torch.stack(log_probs2)
        # 将log_probs2转换为概率分布
        probs2 = torch.exp(log_probs2)
        total_distance = 0.0
        for lp1, p2 in zip(log_probs1, probs2):
          # 计算单个样本之间的KL散度
            kl_div = F.kl_div(lp1.unsqueeze(0), p2.unsqueeze(0), reduction='batchmean', log_target=False)
            total_distance += kl_div.item()

        # 计算平均KL散度
        average_distance = total_distance / len(log_probs1)
        return average_distance

    
    def cosine_distance_accumulators(self,accumulator1, accumulator2):
        """
        计算两个梯度累加器之间的余弦距离。

        :param accumulator1: 第一个梯度累加器。
        :param accumulator2: 第二个梯度累加器。
        :return: 梯度的余弦距离。
        """
        cos_distances = []
        cos = nn.CosineSimilarity(dim=0)

        for name in accumulator1:
            if name in accumulator2:
                grad1 = accumulator1[name].view(-1)
                grad2 = accumulator2[name].view(-1)
            
                # 避免计算空梯度的情况
                if torch.norm(grad1) != 0 and torch.norm(grad2) != 0:
                    cos_sim = cos(grad1, grad2)
                    cos_distances.append(cos_sim.item())

        return sum(cos_distances) / len(cos_distances) if cos_distances else float('nan')