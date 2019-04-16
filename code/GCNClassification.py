import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import GCNConv, GINConv, SGConv
from torch.nn import Linear
import torch.nn as nn

class GCNClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True):
        super(GCNClassification, self).__init__()
        # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = num_classes
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        self.linear_preds = Linear(self.hidden, self.num_classes)
        setattr(self, "conv%d" % 0, GCNConv(self.num_features, self.hidden))
        for i in range(1,config.n_layers):
            setattr(self, "conv%d" % i, GCNConv(self.hidden, self.hidden))
    # forward(x, edge_index, edge_weight=None)
    def forward(self, data):
        
        x, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)
        x = self.linear_preds(x)
        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data):
        self.eval()

        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()

class SGConvClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True):
        super(SGConvClassification, self).__init__()
        # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = num_classes
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        
        setattr(self, "conv%d" % 0, SGConv(self.num_features, self.num_classes,K=config.n_layers))
    # forward(x, edge_index, edge_weight=None)
    def forward(self, data):
        
        x, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
        x = getattr(self, "conv%d" % 0)(x,edge_index)

        if self.graph == True:
            x = scatter_mean(x, data.batch, dim=0)
        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data):
        self.eval()

        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()
        
class SGINClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True):
        super(SGINClassification, self).__init__()
        # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = num_classes
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        
        for i in range(config.n_layers):
            setattr(self, "conv%d" % i, SGConv(self.num_features, self.num_classes,K=i+1))
    # forward(x, edge_index, edge_weight=None)
    def forward(self, data):
        
        hidden_reps=[]
        x0, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
        for i in range(self.n_layers):
            x = getattr(self, "conv%d" % i)(x0,edge_index)
            hidden_reps.append(x)

        output_score = 0
        for i in range(self.n_layers):
            if self.graph == True:
                x = scatter_mean(hidden_reps[i], data.batch, dim=0)
                output_score += x
        return F.log_softmax(output_score, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data):
        self.eval()

        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()        

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
    def forward(self, data):
        hidden_reps=[]
        x, edge_index = torch.ones((len(data.batch),1)).cuda(), data.edge_index
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

    def eval_metric(self, data):
        self.eval()

        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / len(data.y)

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()

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