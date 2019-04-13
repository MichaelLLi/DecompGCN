import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T

from Graph import Graph

class GraphDataset(InMemoryDataset):
    def __init__(self, root, config, 
                    transform=None, pre_transform=None, pre_filter=None):
        self.config = config
        self.num_graphs = config.num_graphs
        transform = getattr(T, config.data_transform)()
        
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        processed_data_file = self.config.dataset_path.split('/')[2] + '.pt'
        return [processed_data_file]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        print('Creating {} new random graphs ... '.format(self.num_graphs))
        data_list = []

        for i in range(self.num_graphs):
            graph = Graph(self.config)
            graph.create_graph()
            data_list.append(graph.data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def max_neighbors(self):
        # Detect maximum number of neighbors
        neighbors = 0
        for i in range(self.__len__()):
            neighbors = max(neighbors, torch.max(self.get(i).y).item())

        self.config.max_neighbors = int(neighbors)


