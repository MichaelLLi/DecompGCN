import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn

from torch_scatter import scatter_mean, scatter_add

from torch_geometric.nn import GINConv, SGConv#, GATConv, GCNConv
from gcnconv_modified import GCNConvModified
from GAT_Modified import GATConv
from SGConv_Modified import SGConv_Modified
#from GraphSage import SAGEConv

import copy
from torch_sparse import spmm, spspmm

from base import GraphClassification


class LConfig:
    pass

def layerconfig(x):
    major_type=x[0]
    level=x[1]
    config=LConfig()
    if major_type=="V":
        config.order=int(level)
        config.normalize=False
        config.edge=False
        config.diag=False
    elif major_type=="E":
        config.order=int(level)
        config.normalize=False
        config.edge=True
        config.diag=True        
    return config
        

class GCNConvModel(GraphClassification):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=False):
        super(GCNConvModel, self).__init__(config, num_classes, graph, classification)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.residual = False
        self.linear_preds = Linear(self.hidden, self.num_classes)
        layertypes=config.layertype.split(",")
        configs=[layerconfig(x) for x in layertypes]
        self.num_layer_types=len(configs)
        for i in range(len(configs)):
            setattr(self, "conv%d%d" % (0,i), GCNConvModified(self.num_features, self.hidden,configs[i]))
            setattr(self, "param%d" % i, torch.nn.Parameter(torch.randn(1)).cuda())           
        for j in range(1,config.n_layers):
            for i in range(len(configs)):
                setattr(self, "conv%d%d" % (j,i), GCNConvModified(self.hidden, self.hidden,configs[i]))

    def forward(self, data, x):
        edge_index = data.edge_index
        #residual_x = torch.zeros((x.shape[0], self.hidden)).to(self.device)
        xs = []
        #x1s, x2s = [], []
        #x1, x2 = x, x
        for i in range(self.n_layers):
#            x = getattr(self, "conv%d" % i)(x, edge_index)
            xo = 0
            for j in range(self.num_layer_types):
                xo = xo + getattr(self, "param%d" % j)*getattr(self, "conv%d%d" % (i,j))(x, edge_index)
                xo = F.leaky_relu(xo, 0.1)
            #x1 = F.leaky_relu(x1, 0.1)
            #x2 = F.leaky_relu(x2, 0.1)

            #x1s.append(x1)
            #x2s.append(x2)
            xs.append(xo)
            x = xo
            #if self.residual == True:
            #    if i == 0:
            #        residual_x = x.clone()
            #    else:
            #        residual_x += x
            #x = F.dropout(x,p=self.dropout_p)
#        x = sum(xs)
        if self.residual == True:
            #x = residual_x.clone()
            x = sum(xs)
        if self.graph == True:
            x = scatter_add(x, data.batch, dim=0)
        out = self.linear_preds(x)
        
        if self.classification == True:
            out = F.log_softmax(out, dim=1)

        return out
       
class SGConvClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True, classification=True, residual=True):
        super(SGConvClassification, self).__init__()
       
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = num_classes
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        
        setattr(self, "conv%d" % 0, SGConv(self.num_features, self.num_classes,K=config.n_layers))
  
    # forward(x, edge_index, edge_weight=None)
    def forward(self, data, x):
       
        edge_index = data.edge_index
        x = getattr(self, "conv%d" % 0)(x,edge_index)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)
        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data, x):
        self.eval()

        _, pred = self.forward(data, x).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()

class SGConvRegression(torch.nn.Module):
    def __init__(self, config, graph=True):
         super(SGConvRegression, self).__init__()
         self.num_features = config.num_features
         self.hidden = config.hidden_units
         self.n_layers=config.n_layers
         self.dropout_p=config.dropout_p
         self.graph = graph
         self.linear_preds = Linear(self.hidden, 1)
         setattr(self, "conv%d" % 0, SGConv(self.num_features, 1,K=config.n_layers))
    
    def forward(self, data, x):
        edge_index = data.edge_index
        x = getattr(self, "conv%d" % 0)(x,edge_index)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)
        return x

    def loss(self, inputs, targets):
        inputs=inputs.float()
        targets=targets.float()
        self.current_loss = F.mse_loss(inputs, targets)
        return self.current_loss

    def eval_metric(self, data, x):
        self.eval()

        pred = self.forward(data, x)
        mse = self.loss(pred,data.y)
        acc = -mse
        return acc  

# class SGConvModifiedRegression(torch.nn.Module):
#     def __init__(self, config, graph=True):
#         super(SGConvModifiedRegression, self).__init__()
#         self.num_features = config.num_features
#         self.hidden = config.hidden_units
#         self.n_layers=config.n_layers
#         self.dropout_p=config.dropout_p
#         self.graph = graph
#         self.linear_preds = Linear(self.hidden, 1)
#         setattr(self, "conv%d" % 0, SGConv_Modified(self.num_features, 1,K=config.n_layers))
#     def forward(self, data):
#         x, edge_index = torch.ones((len(data.batch),1)), data.edge_index
#         x = getattr(self, "conv%d" % 0)(x,edge_index)

#         if self.graph == True:
#             x = scatter_mean(x, data.batch, dim=0)
#         return x

#     def loss(self, inputs, targets):
#         inputs=inputs.float()
#         targets=targets.float()
#         self.current_loss = F.mse_loss(inputs, targets)
#         return self.current_loss

#     def eval_metric(self, data):
#         self.eval()

#         pred = self.forward(data)
#         mse = self.loss(pred,data.y)
#         acc = -mse
#         return acc  

# class SGINClassification(torch.nn.Module):
#     def __init__(self, config, num_classes, graph=True):
#         super(SGINClassification, self).__init__()
#         # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
#         self.num_features = config.num_features
#         self.hidden = config.hidden_units
#         self.num_classes = num_classes
#         self.n_layers=config.n_layers
#         self.dropout_p=config.dropout_p
#         self.graph = graph
        
#         for i in range(config.n_layers):
#             setattr(self, "conv%d" % i, SGConv(self.num_features, self.num_classes,K=i+1))
#     # forward(x, edge_index, edge_weight=None)
#     def forward(self, data):
        
#         hidden_reps=[]
#         x0, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
#         for i in range(self.n_layers):
#             x = getattr(self, "conv%d" % i)(x0,edge_index)
#             hidden_reps.append(x)

#         output_score = 0
#         for i in range(self.n_layers):
#             if self.graph == True:
#                 x = scatter_mean(hidden_reps[i], data.batch, dim=0)
#                 output_score += x
#         return F.log_softmax(output_score, dim=1)

#     def loss(self, inputs, targets):
#         self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
#         return self.current_loss

#     def eval_metric(self, data):
#         self.eval()

#         _, pred = self.forward(data).max(dim=1)
#         correct = pred.eq(data.y).sum().item()
#         acc = correct / len(data.y)

#         return acc

#     def out_to_predictions(self, out):
#         _, pred = out.max(dim=1)
#         return pred

#     def predictions_to_list(self, predictions):
#         return predictions.tolist()        

class GINClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True):
        super(GINClassification, self).__init__()
        # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = num_classes
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        self.linear_preds = Linear(self.hidden, self.num_classes)
        setattr(self, "conv%d" % 0, GINConv(MLP(2,self.num_features, self.hidden, self.hidden)))
        for i in range(1,config.n_layers):
            setattr(self, "conv%d" % i, GINConv(MLP(2,self.hidden, self.hidden, self.hidden)))
    # forward(x, edge_index, edge_weight=None)
    def forward(self, data, x):
        hidden_reps=[]
        edge_index = data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            x = F.relu(x)
            hidden_reps.append(x)
#            x = F.dropout(x,p=self.dropout_p)
        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                x = scatter_mean(hidden_reps[i], data.batch, dim=0)
                x = self.linear_preds(x)
                output_score += x
        return F.log_softmax(output_score, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data, x):
        self.eval()

        _, pred = self.forward(data, x).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()

class GINRegression(torch.nn.Module):
    def __init__(self, config, graph=True):
        super(GINRegression, self).__init__()
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        self.linear_preds = Linear(self.hidden, 1)
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
#            x = F.dropout(x,p=self.dropout_p)
        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                #x = scatter_mean(hidden_reps[i], data.batch, dim=0)
                x = scatter_add(hidden_reps[i], data.batch, dim=0)
                x = self.linear_preds(x)
                output_score += x
        return output_score

    def loss(self, inputs, targets):
        inputs=inputs.float()
        targets=targets.float()
        self.current_loss = F.mse_loss(inputs[:, 0], targets)
        return self.current_loss

    def eval_metric(self, data, x):
        self.eval()

        pred = self.forward(data, x)
        mse = self.loss(pred,data.y)
        acc = -mse
        return acc  
# class GATClassification(torch.nn.Module):
#     def __init__(self, config, num_classes, graph=True):
#         super(GATClassification, self).__init__()
#         self.num_features = config.num_features
#         self.hidden = config.hidden_units
#         self.num_classes = num_classes
#         self.n_layers=config.n_layers
#         self.dropout_p=config.dropout_p
#         self.graph = graph
#         self.linear_preds = Linear(self.hidden, self.num_classes)

#         # GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True)
#         setattr(self, "conv%d" % 0, GATConv(self.num_features, self.hidden, heads=8, concat=False, dropout=self.dropout_p))
#         for i in range(1,config.n_layers):
#             setattr(self, "conv%d" % i, GATConv(self.hidden, self.hidden, heads=8, concat=False, dropout=self.dropout_p))

#     def forward(self, data):
#         x, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
#         for i in range(self.n_layers):
#             x = getattr(self, "conv%d" % i)(x,edge_index)
#             x = F.relu(x)

#         if self.graph == True:
#             x = scatter_mean(x, data.batch, dim=0)

#         x = self.linear_preds(x)
#         return F.log_softmax(x, dim=1)

#     def loss(self, inputs, targets):
#         self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
#         return self.current_loss

#     def eval_metric(self, data):
#         self.eval()

#         _, pred = self.forward(data).max(dim=1)
#         correct = pred.eq(data.y).sum().item()
#         acc = correct / len(data.y)

#         return acc

#     def out_to_predictions(self, out):
#         _, pred = out.max(dim=1)
#         return pred

#     def predictions_to_list(self, predictions):
#         return predictions.tolist()
    
# class GATRegression(torch.nn.Module):
#     def __init__(self, config, graph=True):
#         super(GATRegression, self).__init__()
#         self.num_features = config.num_features
#         self.hidden = config.hidden_units
#         self.n_layers=config.n_layers
#         self.dropout_p=config.dropout_p
#         self.graph = graph
#         self.linear_preds = Linear(self.hidden, 1)

#         # GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True)
#         setattr(self, "conv%d" % 0, GATConv(self.num_features, self.hidden, heads=8, concat=False, dropout=self.dropout_p))
#         for i in range(1,config.n_layers):
#             setattr(self, "conv%d" % i, GATConv(self.hidden, self.hidden, heads=8, concat=False, dropout=self.dropout_p))
#     def forward(self, data):
#         x, edge_index = torch.ones((len(data.batch),1)), data.edge_index
#         for i in range(self.n_layers):
#             x = getattr(self, "conv%d" % i)(x,edge_index)
#             x = F.relu(x)

#         if self.graph == True:
#             x = scatter_mean(x, data.batch, dim=0)
#         x = self.linear_preds(x)
#         return x

#     def loss(self, inputs, targets):
#         inputs=inputs.float()
#         targets=targets.float()
#         self.current_loss = F.mse_loss(inputs, targets)
#         return self.current_loss

#     def eval_metric(self, data):
#         self.eval()

#         pred = self.forward(data)
#         mse = self.loss(pred,data.y)
#         acc = -mse
#         return acc 


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


