import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn.inits import glorot

from torch_scatter import scatter_mean, scatter_add
from torch_sparse import spmm, spspmm

from torch_geometric.nn import GINConv, SGConv, GCNConv
from gcnconv_modified import GCNConvAdvanced
from gat_modified import GATConv
from base import GraphClassification
from utils import LayerConfig, MLP


class GCNConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvModel2, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.residual = config.residual
        self.n_mlp_layers = config.n_mlp_layers
        self.normalize = config.normalize
        layertypes = config.layertype.split(",")
        layer_configs = [self.set_layer_config(layer_type) for layer_type in layertypes]
        
        if config.n_layers>1:
            setattr(self, "conv%d%d" % (0,0), GCNConvAdvanced(self.num_features, self.hidden,layer_configs))
        else:
            setattr(self, "conv%d%d" % (0,0), GCNConvAdvanced(self.num_features, self.num_classes,layer_configs))            
        for j in range(1,config.n_layers-1):
                setattr(self, "conv%d%d" % (j,0), GCNConvAdvanced(self.hidden, self.hidden,layer_configs))
        if self.residual == True and config.n_layers>1:
            setattr(self, "conv%d%d" % (config.n_layers-1,0), GCNConvAdvanced(self.hidden, self.hidden,layer_configs))
        elif config.n_layers>1:
            setattr(self, "conv%d%d" % (config.n_layers-1,0), GCNConvAdvanced(self.hidden, self.num_classes,layer_configs))
            
        
        self.dropout=torch.nn.Dropout(p=self.dropout_p)
        self.linear_preds = Linear(self.hidden, self.num_classes)

    def set_layer_config(self, layer_type):
        major_type, level = layer_type[0], layer_type[1]

        layer_config=LayerConfig()
        layer_config.normalize = self.normalize
        layer_config.order=int(level)
        layer_config.n_mlp_layers=self.n_mlp_layers
        
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
            x = F.leaky_relu(x,0.2)
            xs.append(x)
        
        if self.residual == True:
            x = sum(xs)
        
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
        
        if self.residual == True:
            x = self.linear_preds(x)

        return x
        
        
class GCNConvSimpModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvSimpModel, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.residual = config.residual
        
        self.normalize = config.normalize
        
        setattr(self, "conv%d%d" % (0,0), GCNConv(self.num_features, self.hidden,bias=False))
    
        for j in range(1,config.n_layers-1):
                setattr(self, "conv%d%d" % (j,0), GCNConv(self.hidden, self.hidden,bias=False))
        
        setattr(self, "conv%d%d" % (config.n_layers-1,0), GCNConv(self.hidden, self.hidden,bias=False))
        self.dropout=torch.nn.Dropout(p=self.dropout_p)
        self.linear_preds = Linear(self.hidden, self.num_classes)
    
    def forward(self, data, x):
        edge_index = data.edge_index
        
        xs = []

        for i in range(self.n_layers):
            x = getattr(self, "conv%d%d" % (i,0))(x, edge_index)
            x = self.dropout(getattr(self, "conv%d%d" % (i,0))(x, edge_index))  
            x = F.relu(x)
            xs.append(x)

        if self.residual == True:
            x = sum(xs)    
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
        out = self.linear_preds(x)

        return out


class SGConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(SGConvModel, self).__init__(config, num_classes, graph, classification)

        setattr(self, "conv%d" % 0, SGConv(self.num_features, self.num_classes,K=config.n_layers))

    def forward(self, data, x):    
        edge_index = data.edge_index
        x = getattr(self, "conv%d" % 0)(x,edge_index)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)

        return x


class GINConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(GINConvModel, self).__init__(config, num_classes, graph, classification)
        self.batch_norms = torch.nn.ModuleList()

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(config.n_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(self.num_features, self.num_classes))
            else:
                self.linears_prediction.append(nn.Linear(self.hidden, self.num_classes))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden))

        setattr(self, "conv%d" % 0, GINConv(MLP(config.n_mlp_layers,self.num_features, self.hidden, self.hidden)))
        for i in range(1,config.n_layers):
            setattr(self, "conv%d" % i, GINConv(MLP(config.n_mlp_layers,self.hidden, self.hidden, self.hidden)))

    def forward(self, data, x):
        if x is None:
            x = torch.ones((data.num_nodes, 1)).to(self.device)
        hidden_reps=[]
        hidden_reps.append(x)
        edge_index = data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            hidden_reps.append(x)
            x = F.dropout(x,p=self.dropout_p)

        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                x = scatter_add(hidden_reps[i], data.batch, dim=0)
            else:
                x = hidden_reps[i]
            output_score += self.linears_prediction[i](x)

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
            x = F.leaky_relu(x,0.2)

        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)

        x = self.linear_preds(x)

        if self.classification == True:
            x = F.log_softmax(x, dim=1)

        return x

