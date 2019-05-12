import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_dataset import RandomConnectedGraph, \
						  RandomCliqueGraph, \
        				  RandomTreeCycleGraph, \
        				  RandomTriangleGraph, \
        				  RandomPlanarGraph,\
        				  redditDataset, imdbDataset, proteinDataset, \
        				  CoraDataset, CiteSeerDataset, PubMedDataset, Karate,\
                              QM7bD, QM9D, COLLAB


task_dict = {
    'clique': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomCliqueGraph
    },
    'connectivity': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomConnectedGraph
    },
    'tree_cycle': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomCliqueGraph
    },
    'planar': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomPlanarGraph
    },
    'triangle': {
        'classification': False,
        'graph': True,
        'num_classes': 1,
        'dataset': RandomTriangleGraph
    },
    'imdb-b': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': imdbDataset
    },
    'proteins': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': proteinDataset
    },
    'reddit-b': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': redditDataset
    },
    'karate': {
        'classification': True,
        'graph': False,
        'num_classes': 2,
        'dataset': Karate
    },
    'cora': {
        'classification': True,
        'graph': False,
        'num_classes': 7,
        'dataset': CoraDataset
    },
    'citeseer': {
        'classification': True,
        'graph': False,
        'num_classes': 6,
        'dataset': CiteSeerDataset
    },
    'pubmed': {
        'classification': True,
        'graph': False,
        'num_classes': 3,
        'dataset': PubMedDataset
    },
        'QM7b': {
        'classification': False,
        'graph': True,
        'num_classes': 14,
        'dataset': QM7bD
    },
        'QM9': {
        'classification': False,
        'graph': True,
        'num_classes': 12,
        'dataset': QM9D
    },
    'collab': {
        'classification': True,
        'graph': True,
        'num_classes': 3,
        'dataset': COLLAB
    }     
}


class LayerConfig:
    def __init__(self):
        self.order = 0
        self.normalize = False
        self.edge = False
        self.diag = False  


class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            
            for layer in self.linears:
                layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape).cuda())
                layer.bias = torch.nn.Parameter(torch.ones(layer.weight.shape[1]).cuda())
    
    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
#                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)        
