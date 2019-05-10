from graph_dataset import RandomConnectedGraph, \
						  RandomCliqueGraph, \
        				  RandomTreeCycleGraph, \
        				  RandomTriangleGraph, \
        				  RandomPlanarGraph,\
        				  redditDataset, imdbDataset, proteinDataset, \
        				  CoraDataset, CiteSeerDataset, PubMedDataset


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
        'graph': False,
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
        'dataset': KarateClub
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
    }
}


class LayerConfig:
    def __init__(self):
        self.order = 0
        self.normalize = False
        self.edge = False
        self.diag = False  

        