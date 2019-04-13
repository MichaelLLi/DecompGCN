import torch
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt
import os

class Graph():
    def __init__(self, config, data=None):
        self.data = data
        self.config = config

        self.num_dimensions = config.euclidian_dimensionality
        self.num_features = config.num_features
        self.num_nodes = config.num_nodes
        self.theta_max = config.theta_max
        self.theta_pred = config.theta_pred

    def create_graph(self):
        pos = torch.rand(self.num_nodes, self.num_dimensions)

        edges = []
        y = torch.zeros(self.num_nodes, dtype=torch.long)
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                node1, node2 = pos[i], pos[j]

                if torch.dist(node1, node2) < self.theta_max:
                    edges.append([i, j])
                    edges.append([j, i])

                    if torch.dist(node1, node2) < self.theta_pred:
                        y[i] += 1
                        y[j] += 1

        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0,1)
        x = torch.ones(self.num_nodes, self.num_features)

        # Data(x=None, edge_index=None, edge_attr=None, y=None, pos=None)
        self.data = Data(x=x, edge_index=edge_index, y=y, pos=pos)

    def plot(self):
        g = nx.Graph(incoming_graph_data=self.data.edge_index.transpose(0,1).tolist())
        pos_dict = {}
        # prepare the targets to be displayed

        for i in range(self.data.x.size(0)):
            pos_dict[i] = self.data.x[i].tolist()
            labels_dict[i] = int(self.data.y[i].item())

        self.set_plotting_style()
        nx.draw_networkx(g, pos_dict, labels=labels_dict)
        plt.title("Number of neighbors within euclidian distance {}".format(
            self.config.theta))
        plt.savefig(os.path.join(self.config.temp_dir, 'graph.png'))

    def plot_predictions(self, pred):
        g = nx.Graph(incoming_graph_data=self.data.edge_index.transpose(0,1).tolist())
        pos_dict = {}
        labels_dict = {}

        for i in range(self.data.pos.size(0)):
            pos_dict[i] = self.data.pos[i].tolist()
            labels_dict[i] = '{};{}'.format(int(pred[i]), int(self.data.y[i].item()))

        fig = self.set_plotting_style()
        nx.draw_networkx(g, pos_dict, labels=labels_dict, font_size=10)
        fig.suptitle("Number of neighbors within euclidian distance {}.\nEach node displays 'pred:target'".format(
            self.config.theta_pred))

        img_path = os.path.join(self.config.temp_dir, 'graph_with_predictions.png')
        if os.path.isfile(img_path):
            os.remove(img_path)
        fig.savefig(img_path)
        print('plotted the graph with predictions to {}'.format(img_path))
        return fig

    def set_plotting_style(self):
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('x (euclidian)')
        plt.ylabel('y (euclidian)')

        return fig







