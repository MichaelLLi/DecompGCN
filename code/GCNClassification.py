import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNClassification(torch.nn.Module):
    def __init__(self, config):
        super(GCNClassification, self).__init__()
        # GCNConv(in_channels, out_channels, improved=False, cached=False, bias=True)
        self.num_features = config.num_features
        self.hidden = config.hidden_units
        self.num_classes = config.max_neighbors

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.num_classes+1)

    # forward(x, edge_index, edge_weight=None)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets, reduction='mean')
        return self.current_loss

    def eval_metric(self, data):
        self.eval()

        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes

        return acc

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()