import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
from fedml_api.standalone.dadpfl.sp_functions import conv_fc_condition
from fedml_api.standalone.gossipfl.client import Client
from fedml_api.utils.NetworkTopology import NetworkTopology
import networkx as nx

class GossipFLMutiAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
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
        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )
        self.init_stat_info()

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
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

    def train(self):
        w_global_cifar10 = self.model_trainer[0].get_model_params(self.device)
        w_global_cifar100 = self.model_trainer[1].get_model_params(self.device)
        
        w_per_mdls = [copy.deepcopy(w_global_cifar10) for _ in range(self.args.client_num_in_total // 2)] + \
                    [copy.deepcopy(w_global_cifar100) for _ in range(self.args.client_num_in_total // 2)]
        #w_global = self.model_trainer.get_model_params(self.device)
        #self.stat_info = {"sum_training_flops": 0, "sum_comm_params": 0}
        init_mask_cifar10 = self.generate_random_mask(
            seed=self.args.seed,
            sparsity=self.args.target_sparsity,
            model_dict=self.model_trainer[0].get_model_params(self.device),
        )
        init_mask_cifar100 = self.generate_random_mask(
            seed=self.args.seed,
            sparsity=self.args.target_sparsity,
            model_dict=self.model_trainer[1].get_model_params(self.device),
        )
        w_per_compressed = []
        # w_per_mdls include all the weights of clients (dense model), which is a list of dictionaries.
        # w_per_compressed include all the weights of clients (compressed model, just for communication and will be recovered back),
        #  which is also a list of dictionaries.
        # Initialization
        for clnt in range(self.args.client_num_in_total):
            
            if clnt < self.args.client_num_in_total // 2:
                w_per_compressed_temp = copy.deepcopy(w_global_cifar10)
                for name in init_mask_cifar10.keys():
                    w_per_compressed_temp[name] = w_global_cifar10[name] * init_mask_cifar10[name].to(self.device)
            else:
                w_per_compressed_temp = copy.deepcopy(w_global_cifar100)
                for name in init_mask_cifar100.keys():
                    w_per_compressed_temp[name] = w_global_cifar100[name] * init_mask_cifar100[name].to(self.device)

            w_per_compressed.append(copy.deepcopy(w_per_compressed_temp))

        # Involve tqdm for better visualization
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm

            comm_round_iterable = tqdm(
                range(self.args.comm_round), desc="Comm. Rounds", ncols=100
            )

        for round_idx in comm_round_iterable:
            self.logger.info(
                "################Communication round : {}".format(round_idx)
            )

            client_groups = self._client_random_match(round_idx)
            # Generate random mask for each round, this should be generated in clients with the same seed,
            # as stated in GossipFL paper.
            global_mask_round_idx_cifar10 = self.generate_random_mask(
                seed=round_idx + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer[0].get_model_params(self.device),
            )
            global_mask_round_idx_next_cifar10 = self.generate_random_mask(
                seed=round_idx + 1 + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer[0].get_model_params(self.device),
            )
            global_mask_round_idx_cifar100 = self.generate_random_mask(
                seed=round_idx + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer[1].get_model_params(self.device),
            )
            global_mask_round_idx_next_cifar100 = self.generate_random_mask(
                seed=round_idx + 1 + self.args.seed,
                sparsity=self.args.target_sparsity,
                model_dict=self.model_trainer[1].get_model_params(self.device),
            )
            # Mask generation for each client is only related to the seed, the model_dict is for the structure instruction.

            self.logger.info("client_indexes = " + str(client_groups))
            w_per_compressed_record = copy.deepcopy(w_per_compressed)
            w_per_mdls_record = copy.deepcopy(w_per_mdls)
            tst_results_ths_round = []
            for cur_clnt in range(self.args.client_num_in_total):
                self.logger.info(
                    "@@@@@@@@@@@@@@@@ Training Client CM({}): {}".format(
                        round_idx, cur_clnt
                    )
                )
                # update dataset
                client = self.client_list[cur_clnt]

                # update meta components in personal network
                neighbor_clients = self._detect_neighborhood(client_groups, cur_clnt)
                w_aggr_compress = self._aggregate_compressed_params(
                    cur_clnt,w_per_compressed, neighbor_clients
                )
                #self.logger.info("Dimensions of w_aggr_compress parameters:")
                #for name, mask in w_aggr_compress.items():
                #    self.logger.info(f"{name}: {mask.shape}")
                #self.logger.info("Dimensions of w_per_mdls[cur_clnt] parameters:")
                #for name, mask in w_per_mdls[cur_clnt].items():
                #    self.logger.info(f"{name}: {mask.shape}")
                if cur_clnt < self.args.client_num_in_total // 2:

                    # Each client trains its local dense model, and compresses it to a sparse model for communication.
                    (
                        w_per,
                        w_per_send,
                        test_results,
                    ) = client.train(
                        w_aggr=w_aggr_compress,
                        global_mask=global_mask_round_idx_cifar10,
                        global_mask_next=global_mask_round_idx_next_cifar10,
                        round=round_idx,
                        w_local_last_epoch=copy.deepcopy(w_per_mdls[cur_clnt]),
                    )
                else:
                    #self.logger.info("Dimensions of global_mask_round_idx_cifar100 parameters:")
                    #for name, mask in global_mask_round_idx_cifar100.items():
                    #    self.logger.info(f"{name}: {mask.shape}")
                    #self.logger.info("Dimensions of global_mask_round_idx_next_cifar100 parameters:")
                    #for name, mask in global_mask_round_idx_next_cifar100.items():
                    #    self.logger.info(f"{name}: {mask.shape}")
                    (
                        w_per,
                        w_per_send,
                        test_results,
                    ) = client.train(
                        w_aggr=w_aggr_compress,
                        global_mask=global_mask_round_idx_cifar100,
                        global_mask_next=global_mask_round_idx_next_cifar100,
                        round=round_idx,
                        w_local_last_epoch=copy.deepcopy(w_per_mdls[cur_clnt]),
                    )
                    
                tst_results_ths_round.append(test_results)
                w_per_mdls_record[cur_clnt] = copy.deepcopy(w_per)
                w_per_compressed_record[cur_clnt] = copy.deepcopy(w_per_send)  
            # update communication parameters
            w_per_compressed = copy.deepcopy(w_per_compressed_record)
            w_per_mdls = copy.deepcopy(w_per_mdls_record)
            self._local_test_on_all_clients(tst_results_ths_round, round_idx)

    def _detect_neighborhood(self, clusters_group, client_idx):
        # return a list of neighbor clients
        neighbor_clients = []
        # for cluster in clusters_group:
        #     if client_idx in cluster:
        #         neighbor_clients = cluster
        #         break
        # return neighbor_clients
        for cluster in clusters_group:
            if cluster[0] == client_idx:  # clusters_group 中第一个元素为该客户端本身
                return cluster[1:]  # 返回邻居，不包括客户端自身
        return []

    def _client_random_match(self, round_idx):
        
        # 使用拓扑结构生成图
        graph = self.generate_graph()
        
        # 遍历每个客户端，并使用生成的图结构来确定邻居
        clusters = []
        for client_idx in range(self.args.client_num_in_total):
            # 获取该客户端的邻居（基于生成的图结构）
            neighbors = list(graph.neighbors(client_idx))
            # 添加客户端自身和邻居列表到 clusters
            clusters.append([client_idx] + neighbors)
        
        return clusters

    def generate_random_mask(self, seed, model_dict, sparsity=0.5):
        """
        Generate a random mask for each parameter in the model dictionary with a specific seed.

        Parameters:
        seed (int): The seed for random number generation to ensure reproducibility.
        model_dict (dict): A state dictionary of a DNN model.

        Returns:
        dict: A dictionary containing a random mask for each parameter in the model.
        """
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # If using CUDA, uncomment the following line to ensure reproducibility across multiple GPU calls
        # torch.cuda.manual_seed_all(seed)

        # Create a random mask dictionary with the same structure (prunable) as the model's state_dict
        random_mask_dict = {}
        for key, tensor in model_dict.items():
            if conv_fc_condition(key):
                # Only consider convolutional and fully-connected layers
                # Generate a random mask with the same shape as the model's parameters
                # The mask will have values of 0 or 1. Adjust the probability as needed.
                random_mask = (torch.rand(tensor.size()) > sparsity).float()
                # Save the mask to the dictionary
                random_mask_dict[key] = random_mask

        return random_mask_dict

    def _aggregate_compressed_params(self,clnt_idx, w_locals_compressed, neighbor_clients):
        """
        In GossipFL, they only adopt 2 clients as neighbors, and set Mixing matrix as 0.5 for them.
        So, we set Mixing matrix according to the random match results with mixing matrix value 1/len(neighbor_clients).
        
        :param w_locals_compressed: A list of all local models compressed by each client, sharing a global mask.
        :param neighbor_clients: The list of index neighbor clients
        :return: The aggregated model_compressed
        """
        # Initialize the aggregated model with zeros of the same shape as the first local model
        aggregated_model_compressed = {
            k: torch.zeros_like(v) for k, v in w_locals_compressed[clnt_idx].items()
        }

        # Calculate the mixing matrix value based on the number of neighbor clients
        mixing_matrix_value = 1 / len(neighbor_clients)

        # Aggregate the compressed models from each neighbor
        for neighbor_idx in neighbor_clients:
            neighbor_model_compressed = w_locals_compressed[neighbor_idx]
            for k, v in neighbor_model_compressed.items():
                # Skip aggregation for classifier layers
                if self._is_classifier_layer(k):
                    # For classifier layers, directly copy the value from the first client model
                    if neighbor_idx == clnt_idx:
                        aggregated_model_compressed[k] = v
                else:
                    # For non-classifier layers, perform weighted aggregation
                    aggregated_model_compressed[k] += mixing_matrix_value * v

        return aggregated_model_compressed
    
    def _is_classifier_layer(self, layer_name):
        # 识别分类器层的简单实现，根据实际模型层名称进行调整
        classifier_keywords = ['linear', 'fc', 'classifier']
        for keyword in classifier_keywords:
            if keyword in layer_name:
                return True
            
        return False

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
        self.logger.info(
            "################local_test_on_all_clients after local training in communication round: {}".format(
                round_idx
            )
        )
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        for client_idx in range(self.args.client_num_in_total):
            # test data
            test_metrics["num_samples"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(tst_results_ths_round[client_idx]["test_loss"])
            )

            """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = (
            sum(
                [
                    test_metrics["num_correct"][i] / test_metrics["num_samples"][i]
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )
        test_loss = (
            sum(
                [
                    np.array(test_metrics["losses"][i])
                    / np.array(test_metrics["num_samples"][i])
                    for i in range(self.args.client_num_in_total)
                ]
            )
            / self.args.client_num_in_total
        )

        stats = {"test_acc": test_acc, "test_loss": test_loss}

        self.logger.info(stats)

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
        
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
