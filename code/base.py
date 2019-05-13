import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch.nn import Linear
import torch.nn as nn


class GraphClassification(torch.nn.Module):
    def __init__(self, config, num_classes, graph=True, classification=True):
        super(GraphClassification, self).__init__()
        
        self.num_features = config.num_features
        self.num_classes = num_classes
        self.hidden = config.hidden_units
        self.n_layers=config.n_layers
        self.dropout_p=config.dropout_p
        self.graph = graph
        self.classification = classification

        if self.classification == True:
            self.num_classes = num_classes
    

    def forward(self, data):
        raise NotImplementedError()


    def loss(self, inputs, targets):
        inputs=inputs.float()
        targets=targets.float()

        if self.classification == True:
            #self.current_loss = F.nll_loss(inputs, targets.long(), reduction='mean')
            self.current_loss = F.cross_entropy(inputs, targets.long(), reduction='mean')
        else:
            if len(targets.shape)==1:
                self.current_loss = F.mse_loss(inputs[:, 0], targets)
            else:
                self.current_loss = F.mse_loss(inputs,targets)
        
        return self.current_loss


    def eval_metric(self, data, x):
        self.eval()

        out = self.forward(data, x)
        loss = self.loss(out, data.y)
      
        if self.classification == True:
            _, pred =out.max(dim=1)
            correct = pred.eq(data.y).sum().item()
            acc = correct
        else:
            acc = -loss

        return loss, acc


    def out_to_predictions(self, out):
        if self.classification == True:
            _, pred = out.max(dim=1)
            return pred
        else:
            raise NotImplementedError()


    def predictions_to_list(self, predictions):
        if self.classification == True:
            return predictions.tolist()
        else:
            raise NotImplementedError()


