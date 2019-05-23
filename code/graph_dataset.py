import torch
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Planetoid, QM7b
import os


class RandomTriangleGraph(Dataset):
    def __init__(self, root="../data/triangle/", transform=None, pre_transform=None):
        super(RandomTriangleGraph, self).__init__(root, transform, pre_transform)
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


class RandomFourCyclesGraph(Dataset):
    def __init__(self, root="../data/4_cycle/", transform=None, pre_transform=None):
        super(RandomFourCyclesGraph, self).__init__(root, transform, pre_transform)
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
    

class NCI1(Dataset):
    def __init__(self, root="../data/nci1/", transform=None, pre_transform=None):
        super(NCI1, self).__init__(root, transform, pre_transform)
    
    def __len__(self):
        return 4110

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


def imdbDataset():
    return TUDataset(root="/tmp/imdbb",name="IMDB-BINARY")


def proteinDataset():
    return TUDataset(root="/tmp/proteins",name="PROTEINS")


def CoraDataset():
    return Planetoid(root="/tmp/cora", name='Cora')


def CiteSeerDataset():
    return Planetoid(root="/tmp/citeseer", name='CiteSeer')


def QM7bD():
    return QM7b(root="/tmp/QM7b")
    
