import networkx as nx
from collections import deque
import torch
import torch.nn.functional as F

class GraphTopologyManager:
    def __init__(self, args, device='cpu'):
        """
        Initializes the GraphTopologyManager.

        Parameters:
        - graph: NetworkX graph object representing the network topology.
        - client_list: List of client objects participating in the network.
        - args: Arguments object containing various settings.
        - device: The device ('cpu' or 'cuda') for PyTorch tensors.
        """
        self.args = args
        self.device = device

    def bfs_check_alternative_path(self,graph, node1, node2, max_depth=50):
        """
        Checks for an alternative path between two nodes in the graph using BFS.

        Parameters:
        - node1: The starting node.
        - node2: The target node.
        - max_depth: Maximum depth to search in the graph.

        Returns:
        - True if an alternative path exists, False otherwise.
        """
        visited = {node1}
        queue = deque([(node1, 0)])  # Queue elements are (node, current_depth)
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                break
            
            for neighbor in graph.adj[current_node]:
                if neighbor == node2:
                    return True  # Found an alternative path
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return False

    def add_connection(self,graph, node1, node2):
        """
        Adds an edge between two nodes in the graph.

        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        graph.add_edge(node1, node2)

    def remove_connection(self,graph, node1, node2):
        """
        Removes an edge between two nodes in the graph.

        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        graph.remove_edge(node1, node2)

                            
    def change_neighbors(self,graph,client_list, grap=1, rounds = 0):#grap=1表示图是动态的，grap=0表示图是静态的需要自己改变拓扑邻居节点
        connections_to_remove = set()
        connections_to_add = set()
        for client in client_list:
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
                        if len(graph.adj[client.client_idx]) > client.max_num and client.thres_id < client.max_num:
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
                elif len(graph.adj[client.client_idx]) > client.max_num and client.thres_id < client.max_num:
                    client.thres_id += 1 
            elif client.count_num != 0:
                client.count_num = 0
                    
            
            if client.signal == 1:#需要补充额外的二阶邻居节点
                need_num = client.max_num - len(client.selected_neighbors)
                if len(client.second_neighbors) < need_num:#如果二阶邻居的数量小于差值的话，全部补上
                    for second_id in client.second_neighbors.keys():
                        if client_list[second_id].signal == 1:
                            client_list[second_id].selected_weigh.update({client.client_idx : client.second_neighbors[second_id]})
                            client_list[second_id].selected_neighbors.add(client.client_idx)
                            client.selected_weigh.update({second_id : client.second_neighbors[second_id]})
                            client.selected_neighbors.add(second_id)
                            if grap == 0:#静态图还需要添加拓扑连接
                                add_pair = frozenset([client.client_idx, second_id])
                                connections_to_add.add(add_pair)
                                #self.add_connection(self.graph,client.id,second_id)
                                client.neigh_distance.update({second_id:client.second_neighbors[second_id]})
                                client_list[second_id].neigh_distance.update({client.client_idx: client.second_neighbors[second_id]})
                            if len(client_list[second_id].selected_neighbors) == client_list[second_id].max_num:
                                client_list[second_id].signal = 0#饱和，之后不需要再添加额外的邻居节点
                elif len(client.second_neighbors) >= need_num:#如果二阶邻居的数量大于差值的话，择优补上
                    sorted_second_neighbors = dict(
                        sorted(client.second_neighbors.items(), key=lambda item: item[1])
                    )
                    for second_id in sorted_second_neighbors.keys(): 
                        if client_list[second_id].signal == 1:
                            client_list[second_id].selected_weigh.update({client.client_idx : client.second_neighbors[second_id]})
                            client_list[second_id].selected_neighbors.add(client.client_idx)
                            client.selected_weigh.update({second_id : client.second_neighbors[second_id]})
                            client.selected_neighbors.add(second_id)
                            if grap == 0:#静态图还需要获取拓扑连接
                                #self.add_connection(self.graph,client.id,second_id)
                                add_pair = frozenset([client.client_idx, second_id])
                                connections_to_add.add(add_pair)
                                client.neigh_distance.update({second_id:sorted_second_neighbors[second_id]})
                                client_list[second_id].neigh_distance.update({client.client_idx:sorted_second_neighbors[second_id]})
                            if len(client_list[second_id].selected_neighbors) == client_list[second_id].max_num:
                                client_list[second_id].signal = 0#饱和，不需要再添加额
                            if len(client.selected_neighbors) == client.max_num:#本地也饱和
                                client.signal = 0
                                break#client已经达到了通信瓶颈，现在可以跳到下一个client
                            
                            
                     
        if self.args.step != 2:
            for client in client_list:#尽量选择高质量的邻居节点，如果没有那也只能将就着聚合
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
                            if client_list[neigh_id].signal == 1:
                                client_list[neigh_id].selected_weigh.update({client.client_idx : sorted_neigh_distance[neigh_id]})
                                client.selected_weigh.update({neigh_id : sorted_neigh_distance[neigh_id]})
                                client_list[neigh_id].selected_neighbors.add(client.client_idx)
                                client.selected_neighbors.add(neigh_id)
                                if len(client_list[neigh_id].selected_neighbors) == client_list[neigh_id].max_num:
                                    client_list[neigh_id].signal = 0#饱和，不需要再添加额外的邻居节点
                                if len(client.selected_neighbors) == client.max_num:
                                    client.signal = 0#最后保证所有的客户端都是满员的
                                    break
        else:
            for client in client_list:
                client.thres_id = int(0.8 * client.max_num)
        
        if grap == 0:#必须跳出循环才能进行删除和添加
            for pair in connections_to_add:
                id_list = list(pair)
                id1, id2 = sorted(id_list)
                self.add_connection(graph,id1,id2)    
        
        if grap == 0:#静态图还需要删除拓扑连接
            for client in client_list:
                if client.signal == 0:#不需要补充额外的二阶邻居节点，检查有没有需要删除的节点
                    for neigh_id in graph.adj[client.client_idx]:
                        if neigh_id not in client.selected_neighbors and client.client_idx not in client_list[neigh_id].selected_neighbors:#两边都看不上对方
                            if client_list[neigh_id].signal == 0:
                            #self.remove_connection(self.graph,client.id,neigh_id)
                                remove_pair = frozenset([client.client_idx, neigh_id])
                                connections_to_remove.add(remove_pair)
                                
            for pair in connections_to_remove:
                id_list = list(pair)
                id1, id2 = sorted(id_list)
                self.remove_connection(graph,id1,id2)
                if not self.bfs_check_alternative_path(graph, id1, id2):
                    ######### 如果没有找到替代路径，说明这是割边，恢复边
                    self.add_connection(graph,id1,id2)   
                else:
                    #######如果存在替代路径，安全移除边
                    print(f"Removed edge between {id1} and {id2} safely.")

        # Further code continues as in your provided script, making sure to use self.graph, self.client_list, etc.

    def connect_to_nearest(self,client_list, graph, client_id):
        """
        Connects the client to the nearest clients if it has less than max_num neighbors.

        Parameters:
        - client_id: The ID of the client to connect.
        """
        connections_to_adds = set()
        number = client_list[client_id].max_num - len(graph.adj[client_id])
        for potential_neighbor in client_list:
            if potential_neighbor.client_idx == client_id:
                continue
            if len(graph.adj[potential_neighbor.client_idx]) < client_list[potential_neighbor.client_idx].max_num:
                if not graph.has_edge(potential_neighbor.client_idx, client_id):
                    add_pair = frozenset([client_id, potential_neighbor.client_idx])
                    connections_to_adds.add(add_pair)
                    number -= 1
                    if number == 0:
                        break
        for pair in connections_to_adds:
            id_list = list(pair)
            id1, id2 = sorted(id_list)
            if len(graph.adj[id1]) < client_list[id1].max_num and len(graph.adj[id2]) < client_list[id2].max_num:
                self.add_connection(graph,id1, id2)

    def disconnect_farthest(self, client_id, excess_count):
        """
        Disconnects the farthest neighbors if the client has more than max_num neighbors.

        Parameters:
        - client_id: The ID of the client to disconnect neighbors from.
        - excess_count: Number of excess neighbors to remove.
        """
        farthest_neighbors = sorted(
            self.client_list[client_id].selected_weigh.items(),
            key=lambda item: item[1],
            reverse=True
        )[:excess_count]
        for farthest_neighbor, _ in farthest_neighbors:
            self.remove_connection(client_id, farthest_neighbor)

    def select_neighbors(self,client_list, graph, client):
        """
        Selects neighbors for the client based on KL divergence.

        Parameters:
        - client: The client object.
        """
        selected_neighbors = set()
        selected_weigh = {}
        client_mean = client.running_mean.to(self.device).detach().reshape(1, -1)
        client1_mean_prob = F.softmax(client_mean, dim=1)
        for neigh_id in graph.adj[client.client_idx]:
            other_client_mean = client_list[neigh_id].running_mean.to(self.device).detach().reshape(1, -1)
            client2_mean_prob = F.softmax(other_client_mean, dim=1)
            similarity1 = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
            distance = similarity1.item()
            client.neigh_distance.update({neigh_id: distance})

        sorted_neigh_distance = dict(sorted(client.neigh_distance.items(), key=lambda item: item[1]))
        if client.thres_id < len(sorted_neigh_distance):
            client.threshold = list(sorted_neigh_distance.values())[client.thres_id]

        for neigh_id in sorted_neigh_distance:
            if sorted_neigh_distance[neigh_id] < client.threshold:
                selected_neighbors.add(neigh_id)
                selected_weigh.update({neigh_id: client.neigh_distance[neigh_id]})
                client.num += 1
            if len(selected_neighbors) == client.max_num:
                client.thres_id = client.max_num - 1
                client.threshold = client.neigh_distance[neigh_id]
                break
        client.selected_neighbors = selected_neighbors
        client.selected_weigh = selected_weigh
        if len(selected_neighbors) < client.max_num:
            client.signal = 1

    def get_k_hop_neighbors(self,graph, client_idx, k):
        """
        Gets all k-hop neighbors of a client.

        Parameters:
        - client_idx: Index of the current client.
        - k: The degree of neighbors to retrieve.

        Returns:
        - Set containing all k-hop neighbors.
        """
        if k == 1:
            return set(graph.adj[client_idx])
        else:
            neighbors = set()
            for neighbor in graph.adj[client_idx]:
                neighbors.update(self.get_k_hop_neighbors(graph, neighbor, k - 1))
            neighbors -= set(graph.adj[client_idx])
            neighbors.discard(client_idx)
            return neighbors
    
    def reset_neighbors(self,client_list):
        """
        Resets the neighbors of clients.
        """
        for client in client_list:
            client.signal = 0
            client.num = 0
            client.selected_weigh.clear()#选中节点的余弦距离
            client.selected_neighbors.clear()#选中节点集合
            client.second_neighbors.clear()#二阶邻居节点的余弦距离
            client.neigh_distance.clear()#一阶邻居节点的余弦距离

    def select_neighbors_two(self,graph,client_list, client, k=2):
        """
        Selects neighbors for the client considering two-hop neighbors.

        Parameters:
        - client: The client object.
        - k: The degree of neighbors to consider.
        """
        needed_neighbors = []
        for neigh_id in graph.adj[client.client_idx]:
            if client_list[neigh_id].signal == 1:
                needed_neighbors.append(neigh_id)

        if len(needed_neighbors) > 0:
            for neigh1_id in needed_neighbors:
                client1_mean = client_list[neigh1_id].running_mean.to(self.device).detach().reshape(1, -1)
                client1_mean_prob = F.softmax(client1_mean, dim=1)

                k_hop_neighbors = self.get_k_hop_neighbors(graph, client.client_idx, k)
                for neigh2_id in k_hop_neighbors:
                    if client_list[neigh2_id].signal == 1:
                        if neigh1_id != neigh2_id and neigh2_id not in graph.adj[neigh1_id]:
                            client2_mean = client_list[neigh2_id].running_mean.to(self.device).detach().reshape(1, -1)
                            client2_mean_prob = F.softmax(client2_mean, dim=1)

                            similarity1 = torch.sum(client1_mean_prob * torch.log(client1_mean_prob / client2_mean_prob)) * 1000
                            similarity2 = torch.sum(client2_mean_prob * torch.log(client2_mean_prob / client1_mean_prob)) * 1000

                            similarity1 = similarity1.item()
                            similarity2 = similarity2.item()

                            if similarity1 <= client_list[neigh1_id].threshold and similarity2 <= client_list[neigh2_id].threshold:
                                client_list[neigh1_id].second_neighbors.update({neigh2_id: similarity1})
                                client_list[neigh2_id].second_neighbors.update({neigh1_id: similarity2})
