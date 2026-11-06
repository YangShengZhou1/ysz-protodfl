import copy
import hashlib
import json
import logging
import pickle
import random
import pdb
import numpy as np
import torch
import os
import networkx as nx
from tqdm import tqdm
from fedml_api.standalone.ProToDFL.clients import Client
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon, cosine
from torch import nn
from fedml_api.utils.NetworkTopology import NetworkTopology
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import deque


class ProToDFLAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        dataset = list(dataset)
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset[:8]
        val_data_local_num_dict = dataset[8] if len(dataset) > 8 else None
        val_data_local_dict = dataset[9] if len(dataset) > 9 else None
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_num_dict = val_data_local_num_dict
        self.val_data_local_dict = val_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
            model_trainer, val_data_local_num_dict, val_data_local_dict
        )
        self.init_stat_info()
        self._proto_access_count = 0
        self.comm_sizes = None
        self.cs = args.cs
        self.graph = self.generate_graph()
        self.sub = 0
        self.add = 0
        w_global = self.model_trainer.get_trainable_params(device)
        self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights, tst_results,_ = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, 0)
            self.w_per_globals[c.client_idx] = copy.deepcopy(weights)
            
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
            graph = None
            for _ in range(100):
                candidate = nx.random_geometric_graph(self.args.client_num_in_total, 0.25)
                if nx.is_connected(candidate):
                    graph = candidate
                    break
            if graph is None:
                graph = network_topology.generate_graph('ring')
        return graph
    
    def generate_color_map(self, graph):
        colors = ['red', 'blue', 'green', 'orange', 'gray']
        color_map = ['gray'] * self.args.client_num_in_total
        nodes = sorted(list(graph.nodes()))  # 对节点排序，确保顺序一致
    
        for i in range(len(nodes)):
            color_map[nodes[i]] = colors[(i // 20) % len(colors)]  # 确保每20个节点的颜色一致
    
        return color_map
    
    def draw_graph(self, graph, title, ax, color_map):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color=color_map, ax=ax, node_size=300, font_size=8)
        ax.set_title(title)
        ax.axis('off')

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       model_trainer, val_data_local_num_dict=None, val_data_local_dict=None):
        self.logger.info("############setup_clients (START)#############")
        w_global = self.model_trainer.get_model_params(self.device)
        for client_idx in range(self.args.client_num_in_total):
            c = Client(
                client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger, w_global,
                local_val_data=(val_data_local_dict or {}).get(client_idx),
                local_val_num=(val_data_local_num_dict or {}).get(client_idx, 0),
            )
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        num_comm_params = {}
        self.stat_info["comm"]["oracle_mode"] = self.args.oracle_mode
        self.stat_info["comm"]["oracle_m"] = self.args.oracle_m
        self.stat_info["comm"]["initial_model_sha256"] = self._state_sha256(self.w_per_globals[0])
        self.stat_info["comm"]["data_split_sha256"] = self._data_split_sha256()
        self.logger.info("FINGERPRINT initial_model={} data_split={}".format(
            self.stat_info["comm"]["initial_model_sha256"],
            self.stat_info["comm"]["data_split_sha256"],
        ))
        if self.comm_sizes is None:
            w = self.model_trainer.get_model_params(self.device)
            self.comm_sizes = {
                "model_param_count": sum(v.numel() for v in w.values()),
                "model_bytes_per_client": sum(v.numel() * v.element_size() for v in w.values()),
                "proto_count": int(self.client_list[0].running_mean.numel()),
                "proto_bytes_per_client": int(
                    self.client_list[0].running_mean.numel() * self.client_list[0].running_mean.element_size()
                ),
            }
            self.stat_info["comm"]["model_param_count"] = self.comm_sizes["model_param_count"]
            self.stat_info["comm"]["model_bytes_per_client"] = self.comm_sizes["model_bytes_per_client"]
            self.stat_info["comm"]["proto_count"] = self.comm_sizes["proto_count"]
            self.stat_info["comm"]["proto_bytes_per_client"] = self.comm_sizes["proto_bytes_per_client"]
        for client in self.client_list:#初始化
            client.max_num = len(self.graph.adj[client.client_idx])
        #w_global = self.model_trainer.get_model_params()
        #self.w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        for c in self.client_list:
            #c.model_trainer.set_model_params(w_global)
            weights, _, _ = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, 0)
            self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
        
            #num_comm_params[c.client_idx] = num_comm_params
            #c.neighbor_dict = self._benefit_choose(c.client_idx, self.args.client_num_in_total,
            #                                      self.args.client_num_per_round, self.args.cs)
        #self.graph = self.generate_graph()
        # device = {device} cuda:0apply mask to init weights
        #threshold = 6#逐步缩小阈值 cifar10,resnet18
        #threshold = 86#逐步缩小阈值 cifar100,resnet18
        save_rounds = [0, 5, 10, 50, 100, 300]
        self._save_round_set = {
            int(x) for x in str(self.args.save_models_round).split(",")
            if x.strip().lstrip("-").isdigit()
        }
        graphs_to_save = []
        #threshold = 120#逐步缩小阈值 tiny,resnet18
        #threshold = 6#逐步缩小阈值 cifar10,vgg11
        #threshold = 65#逐步缩小阈值 cifar100,vgg11
        ##########################################################################################################
        ###################################****协议全局训练开始****###############################################/home/LXL/DisPFL-master/fedml_api/standalone/ProToDFL/
        ##########################################################################################################
        for round_idx in tqdm(range(self.args.comm_round+1), desc="Training Progress"):
            if self.args.dynamic == 1:#dynamic topology
                if round_idx % 10 == 0:#per 10 rounds change topology
                    self.graph = self.generate_graph()
                    for client in self.client_list:#初始化
                        client.thres_id = 0#重置阈值


            if round_idx in save_rounds:
                graphs_to_save.append((copy.deepcopy(self.graph), round_idx))
            
            
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
            self._proto_access_count = 0
            
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
    
            for c in self.client_list:
                #if active_ths_rnd[clnt_idx] == 0:#该客户端在本轮训练中不活跃，记录日志并跳过该客户端。
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, str(c.client_idx)))#该客户端在本轮训练中活跃，记录训练日志
                self.logger.info("choose client_indexes: {}, accoring to {}".format(str(self.graph.adj[c.client_idx]), self.args.cs))
                self.logger.info("client.dmax_num {}".format(str(c.max_num)))
                
            
            if self.args.oracle_mode:
                self._select_oracle_neighbors()
            else:
                for client in self.client_list:
                    if self.args.step != 1 :
                        if client.wait_signal != 1:
                            self.select_neighbors(client)
                            self.logger.info("-----------------------最初------------------")
                            self.logger.info('client_id: {}'.format(client.client_idx))
                            self.logger.info('client.neigh_distance: {}'.format(client.neigh_distance))
                            self.logger.info('client.threshold: {}'.format(client.threshold))
                        else:
                            self.compare_exper(client)
                    else:
                        self.compare_exper(client)
                if self.args.step != 2 and self.args.step != 1:
                    for client in self.client_list:
                        self.select_neighbors_two(client, k=self.args.k)#use k-hop neigh message
                if self.args.step != 1:
                    self.change_neighbors(grap=0, rounds = round_idx)
                self._cap_selected_neighbors()
            if round_idx > 0 and round_idx in self._save_round_set:
                self._save_round_models(round_idx)
            self._aggregate_func()#聚合模型
            self._record_comm_round(round_idx)
            for c in self.client_list:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, c.client_idx))
                # update dataset
                # update meta components in personal network
                weights, tst_results,num_comm_param = c.train({k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}, round_idx+1)
                if c.wait_signal == 0:
                    num_comm_param += 256 * len(self.get_k_hop_neighbors(c.client_idx, self.args.k))
                self.w_per_globals[c.client_idx] = {k: v.detach().clone().to(self.device) for k, v in weights.items()}
                num_comm_params[c.client_idx] = num_comm_param
                test_local_metrics = c.local_test(weights, True)
                final_tst_results_ths_round.append(test_local_metrics)            
                tst_results_ths_round.append(tst_results)
                self.stat_info["sum_comm_params"] += num_comm_param
            self.logger.info("sum_comm_params {}".format(self.stat_info["sum_comm_params"]))
            self._test_on_all_clients(round_idx)
            #if threshold >= 40 : #cifar100,vgg11
            
                
        pdf_path = self.draw_and_save_graphs([g for g, r in graphs_to_save], [r for g, r in graphs_to_save])

        comm_path = os.path.join(self.args.output_dir, "logs", self.args.dataset, self.args.identity + "_comm.json")
        os.makedirs(os.path.dirname(comm_path), exist_ok=True)
        with open(comm_path, "w", encoding="utf-8") as comm_file:
            json.dump(self.stat_info["comm"], comm_file, ensure_ascii=False, indent=2)

        print(f"Graphs saved to {pdf_path}")
            # self._local_test_on_all_clients(w_global, round_idx)
        # self.record_avg_inference_flops(w_global)
        
    def draw_and_save_graphs(self, graphs, rounds):
        base_path = os.path.join(self.args.output_dir, "topologies", self.args.identity)
        folder_path = os.path.join(base_path, self.cs)
        os.makedirs(folder_path, exist_ok=True)

        pdf_filename = f"topologies_{self.args.partition_method}_{self.args.k}.pdf"
        pdf_path = os.path.join(folder_path, pdf_filename)
        
        with PdfPages(pdf_path) as pdf:
            fig, axs = plt.subplots(2, 3, figsize=(15, 15))
            axs = axs.flatten()  # Flatten the array of axes for easier indexing
            for i, (graph, round_idx) in enumerate(zip(graphs, rounds)):
                color_map = self.generate_color_map(graph)
                self.draw_graph(graph, f"Topology at Round {round_idx}", axs[i], color_map)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        return pdf_path


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
        self.stat_info["comm"] = {
            "model_param_count": 0,
            "model_bytes_per_client": 0,
            "proto_count": 0,
            "proto_bytes_per_client": 0,
            "rounds": [],
            "total_model_bytes": 0,
            "total_proto_bytes": 0,
            "total_selected_directed_links": 0,
            "total_proto_accesses": 0,
        }

    def _state_sha256(self, state):
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _data_split_sha256(self):
        digest = hashlib.sha256()
        for client in self.client_list:
            for loader in [client.local_training_data, client.local_validation_data, client.local_test_data]:
                if loader is None:
                    digest.update(b"none")
                    continue
                indices = getattr(loader.dataset, "dataidxs", None)
                if indices is None:
                    digest.update(b"all")
                else:
                    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def _save_round_models(self, round_idx):
        """Save same-round local models, prototypes and indices for offline partner tests."""
        save_root = self.args.save_models_dir or os.path.join(self.args.output_dir, "checkpoints")
        round_dir = os.path.join(save_root, self.args.identity, "round_{}".format(round_idx))
        os.makedirs(round_dir, exist_ok=True)

        def indices_of(loader):
            if loader is None:
                return []
            dataidxs = getattr(loader.dataset, "dataidxs", None)
            return [int(x) for x in dataidxs] if dataidxs is not None else []

        metadata = {
            "identity": self.args.identity,
            "round": round_idx,
            "args": vars(self.args),
            "initial_model_sha256": self.stat_info["comm"].get("initial_model_sha256"),
            "data_split_sha256": self.stat_info["comm"].get("data_split_sha256"),
            "graph_edges": [list(map(int, edge)) for edge in self.graph.edges()],
            "clients": [],
        }
        for client in self.client_list:
            model_file = os.path.join(round_dir, "client_{}.pt".format(client.client_idx))
            proto_file = os.path.join(round_dir, "client_{}_proto.pt".format(client.client_idx))
            torch.save(
                {k: v.detach().cpu() for k, v in self.w_per_globals[client.client_idx].items()},
                model_file,
            )
            torch.save(client.running_mean.detach().cpu(), proto_file)
            metadata["clients"].append({
                "client_idx": client.client_idx,
                "train_num": int(client.local_sample_number),
                "val_num": int(client.local_validation_num),
                "test_num": len(indices_of(client.local_test_data)),
                "train_indices": indices_of(client.local_training_data),
                "val_indices": indices_of(client.local_validation_data),
                "test_indices": indices_of(client.local_test_data),
                "selected_neighbors": sorted(int(x) for x in client.selected_neighbors),
                "selected_weigh": {str(k): float(v) for k, v in client.selected_weigh.items()},
                "threshold": float(client.threshold),
                "thres_id": int(client.thres_id),
                "model_file": model_file,
                "proto_file": proto_file,
            })
        with open(os.path.join(round_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        self.logger.info("SAVED round={} checkpoints to {}".format(round_idx, round_dir))

    def _record_comm_round(self, round_idx):
        comm = self.stat_info["comm"]
        directed_links = sum(len(client.selected_neighbors) for client in self.client_list)
        model_bytes = directed_links * comm["model_bytes_per_client"]
        proto_bytes = self._proto_access_count * comm["proto_bytes_per_client"]
        comm["total_model_bytes"] += model_bytes
        comm["total_proto_bytes"] += proto_bytes
        comm["total_selected_directed_links"] += directed_links
        comm["total_proto_accesses"] += self._proto_access_count
        comm["rounds"].append({
            "round": round_idx,
            "edge_count": self.graph.number_of_edges(),
            "selected_directed_links": directed_links,
            "model_bytes": model_bytes,
            "proto_accesses": self._proto_access_count,
            "proto_bytes": proto_bytes,
        })
        self.logger.info(
            "COMM round={} directed_links={} model_bytes={} proto_accesses={} proto_bytes={}".format(
                round_idx, directed_links, model_bytes, self._proto_access_count, proto_bytes
            )
        )

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
    
    
    # def _aggregate_func(self):  # FedAvg平均聚合
    #   aggregate_weight = {}
    #   for c in self.client_list:
    #       aggregate_weight.update({c.client_idx: {k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}})
          
    #   for c in self.client_list:
    #       self.logger.info('Doing local aggregation!')
    #       total_num = self.train_data_local_num_dict[c.client_idx]
    #       for neigh_index in c.selected_neighbors:
    #           total_num += self.train_data_local_num_dict[neigh_index]
          
    #       W = self.train_data_local_num_dict[c.client_idx] / total_num
          
    #       for k in aggregate_weight[c.client_idx].keys():
    #           if not self._is_classifier_layer(k):
    #               aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]
          
    #       for neigh_index in c.selected_neighbors:
    #           w = self.train_data_local_num_dict[neigh_index] / total_num
    #           for k in aggregate_weight[c.client_idx].keys():
    #               if not self._is_classifier_layer(k):
    #                     aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
          
    #   for c in self.client_list:
    #     for k in aggregate_weight[c.client_idx].keys():
    #         if not self._is_classifier_layer(k):
    #             self.w_per_globals[c.client_idx][k] = aggregate_weight[c.client_idx][k]
    
    def _aggregate_func(self):#FedAvg平均聚合
        aggregate_weight = {}
        for c in self.client_list:
            aggregate_weight.update({c.client_idx: {k: v.detach().clone().to(self.device) for k, v in self.w_per_globals[c.client_idx].items()}})
        scope = self.args.aggregate_scope
        self_scale = self.args.self_weight_scale
        for c in self.client_list:
            total_num = self.train_data_local_num_dict[c.client_idx] * self_scale
            for neigh_index in c.selected_neighbors:
                total_num += self.train_data_local_num_dict[neigh_index]
            W = (self.train_data_local_num_dict[c.client_idx] * self_scale) / total_num
            for k in aggregate_weight[c.client_idx].keys():
                if scope == 'base' and self._is_classifier_layer(k):
                    continue
                aggregate_weight[c.client_idx][k] = W * aggregate_weight[c.client_idx][k]#初始化
            for neigh_index in c.selected_neighbors:
                w = self.train_data_local_num_dict[neigh_index] / total_num
                for k in aggregate_weight[c.client_idx].keys():
                    if scope == 'base' and self._is_classifier_layer(k):
                        continue
                    aggregate_weight[c.client_idx][k] += w * self.w_per_globals[neigh_index][k]
        for c in self.client_list:    
            self.w_per_globals[c.client_idx] = aggregate_weight[c.client_idx]   
        
        self.logger.info('Doing local aggregation! scope={} self_scale={}'.format(scope, self_scale))
        
    def _is_classifier_layer(self, layer_name):
        # 识别分类器层的简单实现，根据实际模型层名称进行调整
        classifier_keywords = ['head', 'linear']
        for keyword in classifier_keywords:
            if keyword in layer_name:
                return True
        return False
    
    
    def bfs_check_alternative_path(self,graph, node1, node2, max_depth=50):

        visited = {node1}
        queue = deque([(node1, 0)])  # 队列元素为 (节点, 当前深度)
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                break
            
            for neighbor in graph.adj[current_node]:
                if neighbor == node2:
                    return True  # 找到另一条路径
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return False
    
    def change_neighbors(self, grap=1, rounds = 0):#grap=1表示图是动态的，grap=0表示图是静态的需要自己改变拓扑邻居节点
        connections_to_remove = set()
        connections_to_add = set()
        for client in self.client_list:
            if self.args.step != 3:#跳过停止等待策略
                if rounds >= 100 and self.args.dynamic == 0:
                    if len(client.second_neighbors) == 0 :#没有需要补充的二阶邻居节点，停止邻居选择
                        # if client.thres_id < client.max_num - 1 :#稍微增加阈值，优化拓扑连接网络
                        #     if client.count_num != 0:
                        #         client.count_num += 1
                        #         client.thres_id = min(client.thres_id+client.count_num , client.max_num)
                        #     else:
                        #         client.count_num = 1
                        #         client.thres_id += 1
                        if len(self.graph.adj[client.client_idx]) > client.max_num and client.thres_id < client.max_num:
                            client.thres_id += 1 
                        if client.wait_signal == 0 and client.thres_id >= client.max_num - 1 :#在本轮中无邻居优化，下一轮跳过
                            client.wait_signal = 2
                            client.wait_time[0] = 0
                            client.wait_time[1] = 0
                        elif client.wait_signal == 1 and client.thres_id >= client.max_num - 1 :#在本轮中跳过邻居
                            client.wait_time[0] -= 1
                            if client.wait_time[0] == 0:#最后一轮停止等待
                                client.wait_signal = 2#设置为过渡状态,再观察在下一轮中是否有邻居节点更新
                        elif client.wait_signal == 2 and client.thres_id >= client.max_num - 1 :#还是没有邻居节点更新，延迟等待时间
                            if 2**client.wait_time[1] >16:
                                client.wait_time[0] = min(self.args.t_max,client.wait_time[1]*2)
                                client.wait_time[1] = min(self.args.t_max,client.wait_time[1]*2)
                            else:
                                client.wait_time[0] = 2**client.wait_time[1]
                                client.wait_time[1] = 2**client.wait_time[1]
                            client.wait_signal = 1
                            continue#跳过这个邻居
                    elif client.wait_signal == 2:#出现了新的连接节点更新，等待时间重置
                        client.wait_time[0] = 0
                        client.wait_time[1] = 0
                        client.wait_signal = 0
                    elif client.count_num != 0:
                        client.count_num = 0
                        
                        
                        
            if len(client.second_neighbors) == 0 :#没有需要补充的二阶邻居节点，停止邻居选择
                if client.thres_id < client.max_num - 1 :#稍微增加阈值，优化拓扑连接网络
                    if client.count_num != 0:
                        client.count_num += 1
                        client.thres_id = min(client.thres_id+client.count_num , client.max_num-1)
                    else:
                        client.count_num = 1
                        client.thres_id += 1
                elif len(self.graph.adj[client.client_idx]) > client.max_num and client.thres_id < client.max_num:
                    client.thres_id += 1 
            elif client.count_num != 0:
                client.count_num = 0
                    
            
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
                                self.client_list[second_id].signal = 0#饱和，不需要再添加额
                            if len(client.selected_neighbors) == client.max_num:#本地也饱和
                                client.signal = 0
                                break#client已经达到了通信瓶颈，现在可以跳到下一个client
                            
                            
                     
        if self.args.step != 2:
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
        else:
            for client in self.client_list:
                client.thres_id = int(0.8 * client.max_num)
        
        if grap == 0:#必须跳出循环才能进行删除和添加
            for pair in connections_to_add:
                id_list = list(pair)
                id1, id2 = sorted(id_list)
                self.add_connection(self.graph,id1,id2)    
        
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
                if not self.bfs_check_alternative_path(self.graph, id1, id2):
                    ######### 如果没有找到替代路径，说明这是割边，恢复边
                    self.add_connection(self.graph,id1,id2)   
                else:
                    #######如果存在替代路径，安全移除边
                    print(f"Removed edge between {id1} and {id2} safely.")
                    
    def connect_to_nearest(self, client_id):
        min_distance = float('inf')
        closest_neighbor = None
        connections_to_adds = set()
        number = self.client_list[client_id].max_num - len(self.graph.adj[client_id])
        for potential_neighbor in self.client_list:
            if potential_neighbor.client_idx == client_id:
                continue
            if len(self.graph.adj[potential_neighbor.client_idx]) < self.client_list[potential_neighbor.client_idx].max_num:
                if not self.graph.has_edge(potential_neighbor.client_idx, client_id):
                    #self.add_connection(self.graph, client_id, closest_neighbor)
                    add_pair = frozenset([client_id, potential_neighbor.client_idx])
                    connections_to_adds.add(add_pair)
                    number -= 1
                    if number == 0:
                        break
        for pair in connections_to_adds:
            id_list = list(pair)
            id1, id2 = sorted(id_list)
            if len(self.graph.adj[id1])<self.client_list[id1].max_num and len(self.graph.adj[id2])<self.client_list[id2].max_num:
                self.add_connection(self.graph,id1,id2) 

    def disconnect_farthest(self, client_id, excess_count):
        farthest_neighbors = sorted(
            self.client_list[client_id].selected_weigh.items(),
            key=lambda item: item[1],
            reverse=True
        )[:excess_count]
        for farthest_neighbor, _ in farthest_neighbors:
            self.remove_connection(self.graph, client_id, farthest_neighbor)
                        
            
        
        
        
        
    # def select_neighbors(self, client, similarity_threshold = 2):
    #     selected_neighbors = set()
    #     selected_weigh = {}
    #     client_mean = client.running_mean.to(self.device).detach().reshape(1, -1)
    #     client1_mean_prob = F.softmax(client_mean, dim=1)
    #     for neigh_id in self.graph.adj[client.client_idx]:
    #         if self.client_list[neigh_id].wait_signal == 1:
    #             selected_neighbors.add(neigh_id)#直接默认邻居，不能改变拓扑结构
    #             continue
    #         other_client_mean = self.client_list[neigh_id].running_mean.to(self.device).detach().reshape(1, -1)
    #     # 计算余弦相似度
    #         client2_mean_prob = F.softmax(other_client_mean, dim=1)
    #         distance = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
    #         distance = distance.item()
    #         client.neigh_distance.update({neigh_id : distance})#统计直接相连的邻居的距离
            
    #     sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
    #     for neigh_id in sorted_neigh_distance:
    #         if len(selected_neighbors) == client.max_num:
    #             break
    #         if sorted_neigh_distance[neigh_id] < similarity_threshold:
    #             selected_neighbors.add(neigh_id)
    #             selected_weigh.update({neigh_id : client.neigh_distance[neigh_id]})
    #         else:#以大于阈值的最小的kl散度距离作为衡量是否二跳邻居节点的依据
    #             if client.threshold > sorted_neigh_distance[neigh_id]:
    #                 client.threshold = sorted_neigh_distance[neigh_id]
    #         if len(selected_neighbors) == client.max_num:#不再需要额外的二阶邻居了
    #             client.threshold = client.neigh_distance[neigh_id]
    #             break
    #     if client.threshold ==  500:
    #        client.threshold = similarity_threshold    
    #     client.selected_neighbors = selected_neighbors
    #     client.selected_weigh = selected_weigh
    #     if len(selected_neighbors) < client.max_num:
    #         client.signal = 1#需要额外的二阶节点
    
    def _select_oracle_neighbors(self):
        """Diagnostic upper bound: choose globally closest prototypes.

        oracle_mode=1 selects top-m by one-way prototype KL; oracle_mode=2
        selects all other clients. This method intentionally ignores the base
        graph and discovery cost, so it must never be presented as a runnable
        decentralized method.
        """
        n_clients = len(self.client_list)
        m = n_clients - 1 if self.args.oracle_mode == 2 else min(
            max(1, self.args.oracle_m), n_clients - 1
        )
        probs = [
            F.softmax(c.running_mean.to(self.device).detach().reshape(1, -1), dim=1)
            for c in self.client_list
        ]
        for i, client in enumerate(self.client_list):
            distances = {}
            for j, other in enumerate(self.client_list):
                if i == j:
                    continue
                distance = torch.sum(probs[i] * torch.log(probs[i] / probs[j])) * 1000
                distance = torch.nan_to_num(
                    distance, nan=1e12, posinf=1e12, neginf=-1e12
                ).item()
                distances[other.client_idx] = float(distance)
            selected = sorted(distances, key=distances.get)[:m]
            client.selected_neighbors = set(selected)
            client.selected_weigh = {idx: distances[idx] for idx in selected}
            client.max_num = m
            client.num = len(selected)
            client.signal = 0
        self.logger.info(
            "ORACLE mode={} m={} (global prototype information, no graph/discovery cost)".format(
                self.args.oracle_mode, m
            )
        )

    def _cap_selected_neighbors(self):
        """Keep at most max_neighbors collaborators, ranked by KL distance."""
        m = self.args.max_neighbors
        if m <= 0:
            return
        for client in self.client_list:
            if len(client.selected_neighbors) > m:
                ranked = sorted(
                    client.selected_neighbors,
                    key=lambda idx: client.selected_weigh.get(
                        idx, client.neigh_distance.get(idx, float("inf"))
                    ),
                )
                keep = ranked[:m]
                client.selected_neighbors = set(keep)
                client.selected_weigh = {
                    idx: client.selected_weigh.get(
                        idx, client.neigh_distance.get(idx, float("inf"))
                    )
                    for idx in keep
                }
            client.max_num = m
            client.num = len(client.selected_neighbors)
        self.logger.info("CAP max_neighbors={}".format(m))

    def select_neighbors(self, client):
        selected_neighbors = set()
        selected_weigh = {}
        client_mean = client.running_mean.to(self.device).detach().reshape(1, -1)
        client1_mean_prob = F.softmax(client_mean, dim=1)
        for neigh_id in self.graph.adj[client.client_idx]:
            # if self.client_list[neigh_id].wait_signal == 1:
            #     client.neigh_distance.update({neigh_id : 0})#直接默认邻居，不能改变拓扑结构
            #     continue
            other_client_mean = self.client_list[neigh_id].running_mean.to(self.device).detach().reshape(1, -1)
        # 计算余弦相似度
            client2_mean_prob = F.softmax(other_client_mean, dim=1)
            similarity1 = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
            distance = similarity1.item()
            client.neigh_distance.update({neigh_id : distance})#统计直接相连的邻居的距离
        self._proto_access_count += len(client.neigh_distance)
            
        sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
        if client.thres_id < len(sorted_neigh_distance):
            client.threshold = list(sorted_neigh_distance.values())[client.thres_id]#选择第client.thres_id的距离作为阈值，判断是否有更适合的邻居节点进行选择

        for neigh_id in sorted_neigh_distance:
            if sorted_neigh_distance[neigh_id] < client.threshold:
                selected_neighbors.add(neigh_id)
                selected_weigh.update({neigh_id : client.neigh_distance[neigh_id]})
                client.num += 1
            #else:#以大于阈值的最小的kl散度距离作为衡量是否二跳邻居节点的依据
            #    if client.threshold > sorted_neigh_distance[neigh_id]
            #        client.threshold = sorted_neigh_distance[neigh_id]
            if len(selected_neighbors) == client.max_num:#不再需要额外的二阶邻居了
                client.thres_id = client.max_num - 1 
                client.threshold = client.neigh_distance[neigh_id]
                break 
        client.selected_neighbors = selected_neighbors
        client.selected_weigh = selected_weigh
        if len(selected_neighbors) < client.max_num:
            client.signal = 1#需要额外的二阶节点
            
            
    def get_k_hop_neighbors(self, client_idx, k):
        """Nodes within shortest-path distance ``[1, k]`` from ``client_idx``.

        The search radius is defined on the base communication graph as all
        nodes whose shortest-path distance is at most ``k``, excluding the
        source itself. This replaces the previous recursive implementation,
        which did not guarantee this set for ``k >= 3``.
        """
        if k <= 0:
            return set()
        lengths = nx.single_source_shortest_path_length(self.graph, client_idx)
        return {node for node, dist in lengths.items() if 0 < dist <= k}


    
    def select_neighbors_two(self, client, k=2):
        needed_neighbors = []
        for neigh_id in self.graph.adj[client.client_idx]:
            if self.client_list[neigh_id].signal == 1:
                needed_neighbors.append(neigh_id)  # 统计需要额外邻居节点信息的客户端id号

        if len(needed_neighbors) > 0:
            for neigh1_id in needed_neighbors:
                client1_mean = self.client_list[neigh1_id].running_mean.to(self.device).detach().reshape(1, -1)
                client1_mean_prob = F.softmax(client1_mean, dim=1)

                # 以需要补充邻居的节点 A 自身为搜索起点，取 A 的 k 跳半径内候选
                k_hop_neighbors = self.get_k_hop_neighbors(neigh1_id, k)
                for neigh2_id in k_hop_neighbors:
                    if self.client_list[neigh2_id].signal == 1:
                        if neigh1_id != neigh2_id and neigh2_id not in self.graph.adj[neigh1_id]:  # 邻居条件判断
                            client2_mean = self.client_list[neigh2_id].running_mean.to(self.device).detach().reshape(1, -1)
                            client2_mean_prob = F.softmax(client2_mean, dim=1)
                            self._proto_access_count += 1

                            similarity1 = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
                            similarity2 = torch.sum(client2_mean_prob * torch.log(client2_mean_prob / client1_mean_prob)) * 1000

                            similarity1 = similarity1.item()  # 转换为Python标量
                            similarity2 = similarity2.item()  # 转换为Python标量

                            if similarity1 <= self.client_list[neigh1_id].threshold and similarity2 <= self.client_list[neigh2_id].threshold:
                                self.client_list[neigh1_id].second_neighbors.update({neigh2_id: similarity1})  # 更新邻居信息
                                self.client_list[neigh2_id].second_neighbors.update({neigh1_id: similarity2})

