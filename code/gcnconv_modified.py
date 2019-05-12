import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter_
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot
from torch.nn import Linear

from utils import MLP

def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    loop_index = torch.arange(0,
	                          num_nodes,
	                          dtype=torch.long,
	                          device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
	    assert edge_weight.numel() == edge_index.size(1)
	    loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
	    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_weight

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
    def __init__(self, in_channels, out_channels, configs):
        super(GCNConvAdvanced, self).__init__(in_channels, out_channels)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_configs = len(configs)
        for i in range(self.num_configs):
            setattr(self, "param%d" % i, torch.nn.Parameter(torch.rand(1)-0.5).cuda()) 
        self.configs=configs
        self.mlp = MLP(configs[0].n_mlp_layers, self.in_channels, self.out_channels, self.out_channels).to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for i in range(self.num_configs):
            xs.append(getattr(self, "param%d" % i) * self.forward_layer(self.configs[i], x, edge_index))
            #xs.append(self.forward_layer(self.configs[i], x, edge_index)) 
        x = sum(xs)
#        import pdb
#        pdb.set_trace()
        return self.mlp(x)
            
    def forward_layer(self, config, x, edge_index, edge_weight=None):
        normalize = config.normalize
        order = config.order
        edge = config.edge
        diag = config.diag
        #x = torch.matmul(x, self.weight)
        #mlp = MLP(2, x.shape[1], 32, self.out_channels).to(self.device)        
        #x = mlp(x)        

        if normalize == True:
            if not self.cached or self.cached_result is None:
                edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=None,
                                     device=edge_index.device)
                fill_value = 1 
                num_nodes = x.size(0)
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight, fill_value, num_nodes)

                row, col = edge_index
                deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                
                index = edge_index
                value = torch.ones(edge_index.shape[1]).to(self.device)
                dense_adj = torch.sparse.FloatTensor(index, value).to_dense()    
                
                dense_adj = deg_inv_sqrt * dense_adj * deg_inv_sqrt.view(-1, 1)
                #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                self.cached_result = dense_adj
                
            dense_adj = self.cached_result
        
        n = x.shape[0]
        
        if normalize == False:
            index = edge_index
            value = torch.ones(edge_index.shape[1]).to(self.device)
            dense_adj = torch.sparse.FloatTensor(index, value).to_dense()
            dense_adj = dense_adj + torch.eye(dense_adj.shape[0]).cuda()
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


