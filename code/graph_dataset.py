import torch
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub, QM7b, QM9, GeometricShapes, Entities
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

class COLLAB(Dataset):
    def __init__(self, root="../data/collab/", transform=None, pre_transform=None):
        super(COLLAB, self).__init__(root, transform, pre_transform)
    
    def __len__(self):
        return 5000

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

class MUTAG(Dataset):
    def __init__(self, root="../data/mutag/", transform=None, pre_transform=None):
        super(MUTAG, self).__init__(root, transform, pre_transform)
    
    def __len__(self):
        return 188

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

class PTC_MR(Dataset):
    def __init__(self, root="../data/ptc_mr/", transform=None, pre_transform=None):
        super(PTC_MR, self).__init__(root, transform, pre_transform)
    
    def __len__(self):
        return 344

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

def GeometricShapesDataset():
    d =  GeometricShapes(root="/tmp/GeometricShapes", pre_transform=T.FaceToEdge).data
    return T.FaceToEdge().__call__(d)

#def MUTAG():
#    d =  Entities(root="/tmp/MUTAG", name="MUTAG")
#    return d
    
