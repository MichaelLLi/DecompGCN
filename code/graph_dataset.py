import torch
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub, QM7b, QM9
import os

class RandomConnectedGraph(Dataset):
    def __init__(self, root="../data/connected/", transform=None, pre_transform=None):
        super(RandomConnectedGraph, self).__init__(root, transform, pre_transform)
    def __len__(self):
        return 1000

    def get(self, idx):
        lev=int(idx/500)+1
        idt=idx % 500
        data = torch.load(self.root + '/connect_{}_{}.pt'.format(lev,idt))
        return data

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass


class RandomCliqueGraph(Dataset):
    def __init__(self, root="../data/clique/", transform=None, pre_transform=None):
        super(RandomCliqueGraph, self).__init__(root, transform, pre_transform)
    def __len__(self):
        return 1000

    def get(self, idx):
        lev=int(idx/500)+3
        idt=idx % 500
        data = torch.load(self.root + '/clique_{}_{}.pt'.format(lev,idt))
        return data

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass


class RandomTreeCycleGraph(Dataset):
    def __init__(self, root="../data/tree_cycle/", transform=None, pre_transform=None):
        super(RandomTreeCycleGraph, self).__init__(root, transform, pre_transform)
    def __len__(self):
        return 1000

    def get(self, idx):
        data = torch.load(self.root + '/' + os.listdir(self.root)[idx])
        return data

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass

class RandomTriangleGraph(Dataset):
    def __init__(self, root="../data/triangle/", transform=None, pre_transform=None):
        super(RandomTriangleGraph, self).__init__(root, transform, pre_transform)
    def __len__(self):
        return 500

    def get(self, idx):
        data = torch.load(self.root + '/' + os.listdir(self.root)[idx])
        return data

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass
    
class RandomPlanarGraph(Dataset):
    def __init__(self, root="../data/planar/", transform=None, pre_transform=None):
        super(RandomPlanarGraph, self).__init__(root, transform, pre_transform)
    
    def __len__(self):
        return 2102

    def get(self, idx):
        data = torch.load(self.root + '/' + os.listdir(self.root)[idx])
        return data

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass

def redditDataset():
    return TUDataset(root="/tmp/redditb",name="REDDIT-BINARY")

def imdbDataset():
    return TUDataset(root="/tmp/imdbb",name="IMDB-BINARY")

def proteinDataset():
    return TUDataset(root="/tmp/proteins",name="PROTEINS")

def CoraDataset():
    return Planetoid(root="/tmp/cora", name='Cora')

def CiteSeerDataset():
    return Planetoid(root="/tmp/citeseer", name='CiteSeer')

def PubMedDataset():
    return Planetoid(root="/tmp/pubmed", name='PubMed')
    
def Karate():
    return KarateClub()

def QM7bD():
    return QM7b(root="/tmp/QM7b")
    
def QM9D():
    return QM9(root="/tmp/QM9")

