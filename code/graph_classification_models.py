import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn

from torch_scatter import scatter_mean, scatter_add
from torch_sparse import spmm, spspmm

from torch_geometric.nn import GINConv, SGConv, GCNConv
from gcnconv_modified import GCNConvModified, GCNConvAdvanced
from GAT_Modified import GATConv
from SGConv_Modified import SGConv_Modified
#from GraphSage import SAGEConv
from base import GraphClassification
from utils import LayerConfig


class GCNConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvModel, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.residual = config.residual
        
        self.normalize = config.normalize
        layertypes = config.layertype.split(",")
        layer_configs = [self.set_layer_config(layer_type) for layer_type in layertypes]
        self.num_layer_types = len(layer_configs)
        
        for i in range(len(layer_configs)):
            setattr(self, "conv%d%d" % (0,i), GCNConvModified(self.num_features, self.hidden,layer_configs[i]))
            setattr(self, "param%d" % i, torch.nn.Parameter(torch.randn(1)).cuda())           
        
        for j in range(1,config.n_layers-1):
            for i in range(len(layer_configs)):
                setattr(self, "conv%d%d" % (j,i), GCNConvModified(self.hidden, self.hidden,layer_configs[i]))
        
        for i in range(len(layer_configs)):
            setattr(self, "conv%d%d" % (config.n_layers-1,i), GCNConvModified(self.hidden, self.num_classes,layer_configs[i]))        
        
        self.dropout=torch.nn.Dropout(p=self.dropout_p)
        #self.linear_preds = Linear(self.hidden, self.num_classes)

    def set_layer_config(self, layer_type):
        major_type, level = layer_type[0], layer_type[1]

        layer_config=LayerConfig()
        layer_config.normalize = self.normalize
        layer_config.order=int(level)
        
        if major_type=="V":
            layer_config.edge=False
            layer_config.diag=False
        elif major_type=="E":
            layer_config.edge=True
            layer_config.diag=True        
        
        return layer_config
    
    def forward(self, data, x):
        edge_index = data.edge_index
        
        xs = []
        layer_xs = []

        for i in range(self.n_layers):
            layer_xs = []
            for j in range(self.num_layer_types):
                weight = getattr(self, "param%d" % j)
                layer_xs.append(weight * self.dropout(getattr(self, "conv%d%d" % (i,j))(x, edge_index)))
            
            x = sum(layer_xs)
            x = F.leaky_relu(x,0.1)
            xs.append(x)

        if self.residual == True:
            x = sum(xs)
        
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
#        out = self.linear_preds(x)
        
        if self.classification == True:
            x = F.log_softmax(x, dim=1)

        return x

class GCNConvModel2(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvModel2, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.residual = config.residual
        
        self.normalize = config.normalize
        layertypes = config.layertype.split(",")
        layer_configs = [self.set_layer_config(layer_type) for layer_type in layertypes]
        
        setattr(self, "conv%d%d" % (0,0), GCNConvAdvanced(self.num_features, self.hidden,layer_configs))
        
        for j in range(1,config.n_layers-1):
                setattr(self, "conv%d%d" % (j,0), GCNConvAdvanced(self.hidden, self.hidden,layer_configs))
        
        setattr(self, "conv%d%d" % (config.n_layers-1,0), GCNConvAdvanced(self.hidden, self.num_classes,layer_configs))        
        
        self.dropout=torch.nn.Dropout(p=self.dropout_p)
        #self.linear_preds = Linear(self.hidden, self.num_classes)

    def set_layer_config(self, layer_type):
        major_type, level = layer_type[0], layer_type[1]

        layer_config=LayerConfig()
        layer_config.normalize = self.normalize
        layer_config.order=int(level)
        
        if major_type=="V":
            layer_config.edge=False
            layer_config.diag=False
        elif major_type=="E":
            layer_config.edge=True
            layer_config.diag=True        
        
        return layer_config
    
    def forward(self, data, x):
        if x is None:
            x = torch.ones((data.num_nodes, 1)).to(self.device)
        edge_index = data.edge_index
        
        xs = []

        for i in range(self.n_layers):
            x = self.dropout(getattr(self, "conv%d%d" % (i,0))(x, edge_index))
            x = F.leaky_relu(x,0.1)
            xs.append(x)

        if self.residual == True:
            x = sum(xs)
        
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
#        out = self.linear_preds(x)
        #if self.classification == True:
            #x = F.log_softmax(x, dim=1)

        return x
        
        
class GCNConvSimpModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvSimpModel, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.residual = config.residual
        
        self.normalize = config.normalize
        
        setattr(self, "conv%d%d" % (0,0), GCNConv(self.num_features, self.hidden))
    
        for j in range(1,config.n_layers-1):
                setattr(self, "conv%d%d" % (j,0), GCNConv(self.hidden, self.hidden))
        
        setattr(self, "conv%d%d" % (config.n_layers-1,0), GCNConv(self.hidden, self.num_classes))
        self.dropout=torch.nn.Dropout(p=self.dropout_p)
        #self.linear_preds = Linear(self.hidden, self.num_classes)
    
    def forward(self, data, x):
        edge_index = data.edge_index
        
        xs = []

        for i in range(self.n_layers):
            x = getattr(self, "conv%d%d" % (i,0))(x, edge_index)
            #x = self.dropout(getattr(self, "conv%d%d" % (i,0))(x, edge_index))
            #x = F.leaky_relu(x,0.1)
            xs.append(x)

        if self.residual == True:
            x = sum(xs)
        
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
#        out = self.linear_preds(x)
        
        #if self.classification == True:
            #x = F.log_softmax(x, dim=1)
        return x


class SGConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(SGConvModel, self).__init__(config, num_classes, graph, classification)

        setattr(self, "conv%d" % 0, SGConv(self.num_features, self.num_classes,K=config.n_layers))

    def forward(self, data, x):    
        edge_index = data.edge_index
        x = getattr(self, "conv%d" % 0)(x,edge_index)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)


        if self.classification == True:
            x = F.log_softmax(x, dim=1)

        return x


class GINConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(GINConvModel, self).__init__(config, num_classes, graph, classification)

        self.linear_preds = Linear(self.hidden, self.num_classes)
        setattr(self, "conv%d" % 0, GINConv(MLP(2,self.num_features, self.hidden, self.hidden)))
        for i in range(1,config.n_layers):
            setattr(self, "conv%d" % i, GINConv(MLP(2,self.hidden, self.hidden, self.hidden)))

    def forward(self, data, x):
        hidden_reps=[]
        edge_index = data.edge_index

        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            #x = F.relu(x)
            hidden_reps.append(x)
#           x = F.dropout(x,p=self.dropout_p)
        
        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                x = scatter_add(hidden_reps[i], data.batch, dim=0)
                x = self.linear_preds(x)
                output_score += x

        if self.classification == True:
            output_score = F.log_softmax(output_score, dim=1)

        return output_score


class SGINConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(SGINConvModel, self).__init__(config, num_classes, graph, classification)

        for i in range(config.n_layers):
            setattr(self, "conv%d" % i, SGConv(self.num_features, self.num_classes,K=i+1))

    def forward(self, data, x):       
        hidden_reps=[]
        edge_index = data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            hidden_reps.append(x)

        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                x = scatter_mean(hidden_reps[i], data.batch, dim=0)
                output_score += x
        
        if self.classification == True:
            output_score = F.log_softmax(output_score, dim=1)

        return output_score


class GATConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(GATConvModel, self).__init__(config, num_classes, graph, classification)

        self.linear_preds = Linear(self.hidden, self.num_classes)
        setattr(self, "conv%d" % 0, GATConv(self.num_features, self.hidden, heads=8, concat=False, dropout=self.dropout_p))
        for i in range(1,config.n_layers):
            setattr(self, "conv%d" % i, GATConv(self.hidden, self.hidden, heads=8, concat=False, dropout=self.dropout_p))

    def forward(self, data, x):
        edge_index = data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            x = F.relu(x)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)

        x = self.linear_preds(x)

        if self.classification == True:
            x = F.log_softmax(x, dim=1)

        return x


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

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


# class GraphSageRegression(torch.nn.Module):
#     def __init__(self, config, graph=True):
#         super(GraphSageRegression, self).__init__()
#         self.num_features = config.num_features
#         self.hidden = config.hidden_units
#         self.n_layers=config.n_layers
#         self.dropout_p=config.dropout_p
#         self.graph = graph
#         self.linear_preds = Linear(self.hidden, 1)

#         # GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True)
#         setattr(self, "conv%d" % 0, SAGEConv(self.num_features, self.hidden))
#         for i in range(1,config.n_layers):
#             setattr(self, "conv%d" % i, SAGEConv(self.hidden, self.hidden))
#     def forward(self, data):
#         x, edge_index = torch.randn((len(data.batch),1)), data.edge_index
#         for i in range(self.n_layers):
#             x = getattr(self, "conv%d" % i)(x,edge_index)
#             x = F.leaky_relu(x, 0.1)
#             #x = F.dropout(x,p=self.dropout_p)

#         if self.graph == True:
#             x = scatter_mean(x, data.batch, dim=0)
#         x = self.linear_preds(x)
#         return x

#     def loss(self, inputs, targets):
#         inputs=inputs.float()
#         targets=targets.float()
#         self.current_loss = F.mse_loss(inputs.transpose(0, 1), targets)
#         return self.current_loss

#     def eval_metric(self, data):
#         self.eval()

#         pred = self.forward(data)
#         mse = self.loss(pred,data.y)
#         acc = -mse
#         return acc 
