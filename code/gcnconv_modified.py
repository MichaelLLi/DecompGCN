import torch
from torch_geometric.nn import GCNConv


class GCNConvModified(GCNConv):
    def __init__(self, in_channels, out_channels,config):
        super(GCNConvModified, self).__init__(in_channels, out_channels)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = config.normalize
        self.order = config.order
        self.edge = config.edge
        self.diag = config.diag

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.normalize == True:
            if not self.cached or self.cached_result is None:
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                             self.improved, x.dtype)
                self.cached_result = edge_index, norm
            edge_index, norm = self.cached_result

        n = x.shape[0]

        index = edge_index
        value = torch.ones(edge_index.shape[1]).to(self.device)
        dense_adj = torch.sparse.FloatTensor(index, value).to_dense()
        dense_adj_k = dense_adj

        i = 1
        while i < self.order:
            dense_adj_k = torch.mm(dense_adj_k, dense_adj)
            i += 1

        dense_adj_2k = torch.mm(dense_adj_k, dense_adj_k)

        if self.edge == True:
            # order * 2 + 1
            dense_adj_k = torch.mm(dense_adj_2k, dense_adj)
            return torch.mm(torch.eye(n).to(self.device) * dense_adj_k, x)
        else:
            # order, order * 2
            if self.diag == True:
                dense_adj_k2 = dense_adj_2k
                return torch.mm(torch.eye(n).to(self.device) * dense_adj_k2, x)

            return  torch.mm(dense_adj_k, x)
            

class GCNConvAdvanced(GCNConv):
    def __init__(self, in_channels, out_channels,configs):
        super(GCNConvAdvanced, self).__init__(in_channels, out_channels)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_configs = len(configs)
        for i in range(self.num_configs):
            setattr(self, "param%d" % i, torch.nn.Parameter(torch.randn(1)).cuda()) 
        self.configs=configs

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for i in range(self.num_configs):
            xs.append(getattr(self, "param%d" % i) * self.forward_layer(self.configs[i], x, edge_index))
        
        x = sum(xs)
        return x
            
    def forward_layer(self, config, x, edge_index, edge_weight=None):
        normalize = config.normalize
        order = config.order
        edge = config.edge
        diag = config.diag
        x = torch.matmul(x, self.weight)

        if normalize == True:
            if not self.cached or self.cached_result is None:
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                             self.improved, x.dtype)
                self.cached_result = edge_index, norm
            edge_index, norm = self.cached_result

        n = x.shape[0]

        index = edge_index
        value = torch.ones(edge_index.shape[1]).to(self.device)
        dense_adj = torch.sparse.FloatTensor(index, value).to_dense()
        dense_adj_k = dense_adj

        i = 1
        while i < order:
            dense_adj_k = torch.mm(dense_adj_k, dense_adj)
            i += 1

        dense_adj_2k = torch.mm(dense_adj_k, dense_adj_k)

        if edge == True:
            # order * 2 + 1
            dense_adj_k = torch.mm(dense_adj_2k, dense_adj)
            return torch.mm(torch.eye(n).to(self.device) * dense_adj_k, x)
        else:
            # order, order * 2
            if diag == True:
                dense_adj_k2 = dense_adj_2k
                return torch.mm(torch.eye(n).to(self.device) * dense_adj_k2, x)

            return  torch.mm(dense_adj_k, x)


