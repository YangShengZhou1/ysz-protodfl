import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class NetworkTopology:
    def __init__(self, client_num_in_total):
        self.client_num_in_total = client_num_in_total

    def generate_graph(self, topology_type):
        if topology_type == 'ring':
            return self._generate_randomized_ring_topology()
        elif topology_type == 'grid':
            return self._generate_randomized_grid_topology()
        elif topology_type == 'exp':
            return self._generate_randomized_exp_topology()
        elif topology_type == 'full':
            return self._generate_randomized_full_topology()
        elif topology_type == 'random':
            return self._generate_randomized_10_neighbors_topology()
        else:
            raise ValueError("Invalid topology type")

    def _generate_randomized_nodes(self):
        nodes = list(range(self.client_num_in_total))
        random.shuffle(nodes)
        return nodes

    def _generate_randomized_ring_topology(self):
        nodes = self._generate_randomized_nodes()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i in range(self.client_num_in_total):
            graph.add_edge(nodes[i], nodes[(i + 1) % self.client_num_in_total])
        return graph

    def _generate_randomized_grid_topology(self):
        nodes = self._generate_randomized_nodes()
        side_length = int(np.ceil(np.sqrt(self.client_num_in_total)))
        
        # Create grid
        graph = nx.grid_2d_graph(side_length, side_length)
        
        # Ensure graph has correct number of nodes by trimming excess nodes
        if len(graph.nodes()) > self.client_num_in_total:
            excess_nodes = len(graph.nodes()) - self.client_num_in_total
            for node in list(graph.nodes())[-excess_nodes:]:
                graph.remove_node(node)
        
        # Relabel nodes
        mapping = {node: nodes[i] for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)
        return graph

    def _generate_randomized_exp_topology(self):
        for _ in range(100):
            graph = nx.random_geometric_graph(self.client_num_in_total, 0.25)
            if nx.is_connected(graph):
                return graph
        return self._generate_randomized_ring_topology()
    
    def _generate_randomized_full_topology(self):
        nodes = self._generate_randomized_nodes()
        graph = nx.complete_graph(nodes)
        return graph
    
    
    def _generate_randomized_10_neighbors_topology(self):
        nodes = self._generate_randomized_nodes()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        if self.client_num_in_total <= 10:
            return nx.complete_graph(nodes)

        for i in range(self.client_num_in_total):
            for j in range(1, 6):
                neighbor = nodes[(i + j) % self.client_num_in_total]
                if not graph.has_edge(nodes[i], neighbor):
                    graph.add_edge(nodes[i], neighbor)

        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            while len(components) > 1:
                component1 = random.choice(components)
                component2 = random.choice(components)
                node1 = random.choice(list(component1))
                node2 = random.choice(list(component2))
                graph.add_edge(node1, node2)
                components = list(nx.connected_components(graph))

        target_degree = min(10, self.client_num_in_total - 1)
        for node in nodes:
            while len(list(graph.neighbors(node))) < target_degree:
                potential_neighbors = set(nodes) - set(graph.neighbors(node)) - {node}
                if not potential_neighbors:
                    break
                neighbor = random.choice(list(potential_neighbors))
                graph.add_edge(node, neighbor)

        return graph

    def remove_connection(self, graph, client_id1, client_id2):
        if graph.has_edge(client_id1, client_id2):
            graph.remove_edge(client_id1, client_id2)

    def add_connection(self, graph, client_id1, client_id2):
        if client_id1 in graph.nodes() and client_id2 in graph.nodes():
            if not graph.has_edge(client_id1, client_id2):
                graph.add_edge(client_id1, client_id2)
                
    # Function to generate color map
    def generate_color_map(self, graph):
        colors = ['red', 'blue', 'green', 'orange', 'gray']  # 选择五种颜色
        color_map = ['purple'] * self.client_num_in_total  # Default color
        nodes = list(graph.nodes())
        
        for i in range(len(nodes)):
            color_map[nodes[i]] = colors[i // 20 % len(colors)]

        return color_map
    # Define function to draw graphs with improved aesthetics
    def draw_graph(self,graph, layout, title, ax, color_map):
        pos = layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color=color_map, ax=ax, node_size=300, font_size=8)
        ax.set_title(title)
        ax.axis('off')  # Hide the axis


# Example usage:
#network_topology = NetworkTopology(client_num_in_total=100)

#randomized_ring_graph = network_topology.generate_graph('ring')
#randomized_grid_graph = network_topology.generate_graph('grid')
#randomized_exp_graph = network_topology.generate_graph('exp')
#randomized_full_graph = network_topology.generate_graph('full')

# Save graphs to PDF
#pdf_path = "/home/amax/check/lxl/LXL/DisPFL-master/results/topologies.pdf"

#with PdfPages(pdf_path) as pdf:
#    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

#    network_topology.draw_graph(randomized_ring_graph, nx.circular_layout, "Randomized Ring Topology", axs[0, 0], network_topology.generate_color_map(randomized_ring_graph))
#    network_topology.draw_graph(randomized_grid_graph, nx.spring_layout, "Randomized Grid Topology", axs[0, 1], network_topology.generate_color_map(randomized_grid_graph))
#    network_topology.draw_graph(randomized_exp_graph, nx.spring_layout, "Randomized Exponential Topology", axs[1, 0], network_topology.generate_color_map(randomized_exp_graph))
#    network_topology.draw_graph(randomized_full_graph, nx.shell_layout, "Randomized Full Topology", axs[1, 1], network_topology.generate_color_map(randomized_full_graph))

#    plt.tight_layout()
##   pdf.savefig(fig)
#    plt.close(fig)


#print(f"Graphs saved to {pdf_path}")
