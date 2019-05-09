import torch
from torch_geometric.data import DataLoader

import os
from tensorboardX import SummaryWriter
import shutil

#from Graph import Graph
from graph_dataset import RandomConnectedGraph, RandomCliqueGraph, \
        RandomTreeCycleGraph, RandomTriangleGraph, RandomPlanarGraph,\
        redditDataset, imdbDataset, proteinDataset, CoraDataset, CiteSeerDataset, PubMedDataset
from config import Config
from graph_classification_models import GINClassification, SGConvClassification, \
         GINRegression,  SGConvRegression#, SGConvModifiedRegression
from graph_classification_models import GCNConvModel
from random import shuffle
from torch_geometric.datasets import KarateClub

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_dict = {
    'clique': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomCliqueGraph
    },
    'connectivity': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomConnectedGraph
    },
    'tree_cycle': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomCliqueGraph
    },
    'planar': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': RandomPlanarGraph
    },
    'triangle': {
        'classification': False,
        'graph': True,
        'num_classes': 1,
        'dataset': RandomTriangleGraph
    },
    'imdb-b': {
        'classification': True,
        'graph': False,
        'num_classes': 2,
        'dataset': imdbDataset
    },
    'proteins': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': proteinDataset
    },
    'reddit-b': {
        'classification': True,
        'graph': True,
        'num_classes': 2,
        'dataset': redditDataset
    },
    'karate': {
        'classification': True,
        'graph': False,
        'num_classes': 2,
        'dataset': KarateClub
    },
    'cora': {
        'classification': True,
        'graph': False,
        'num_classes': 7,
        'dataset': CoraDataset
    },
    'citeseer': {
        'classification': True,
        'graph': False,
        'num_classes': 6,
        'dataset': CiteSeerDataset
    },
    'pubmed': {
        'classification': True,
        'graph': False,
        'num_classes': 3,
        'dataset': PubMedDataset
    }
}

def load_data(config):
    print("Task: ", config.task)

    task = config.task
    dataset = task_dict[task]['dataset']()
    graph = task_dict[task]['graph']

    if task not in  ['reddit-b', 'proteins', 'imdb-b', \
                        'cora', 'citeseer', 'pubmed']:
        data_list=[]
        for i in range(dataset.__len__()):
            data_list.append(dataset[i])
    else:
        data_list=dataset
    
    if graph == True:
        train_idx = int(len(data_list) * (1 - config.test_split -
                                            config.validation_split))
        valid_idx = int(len(data_list) * (1 - config.test_split))
        if task not in  ['reddit-b', 'proteins', 'imdb-b', \
                             'cora', 'citeseer', 'pubmed']:
            shuffle(data_list)
        else:
            data_list = data_list.shuffle()
        
        train_dataset = data_list[:train_idx]
        valid_dataset = data_list[train_idx:valid_idx]
        test_dataset = data_list[valid_idx:]

        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size_train, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size_train, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size_eval, shuffle=True)

        return train_dataset, valid_dataset, train_loader, valid_loader, test_loader
    else:
        data = dataset[0]
        
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.train_mask[:data.num_nodes - 1000] = 1
        data.val_mask = None
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.test_mask[data.num_nodes - 500:] = 1
        
        return data, None, None, None, None

def load_model(device, config):
    print("Model: ", config.model)

    task = config.task
    graph = task_dict[task]['graph']
    num_classes = task_dict[task]['num_classes']
    classification = task_dict[task]['classification']

    if config.model == 'GCN':
        model = GCNConvModel(config, num_classes, graph=graph, classification=classification, residual=config.residual)
    elif config.model == "GIN":
        if classification == True:
            model = GINClassification(config, 2)
        else:
            model = GINRegression(config, 1)
    elif config.model == "SGConv":
        if classification == True:
            model = SGConvClassification(config, 2)
        else:
            model = SGConvRegression(config, 1)
    # elif config.model == "SGIN":
    #     model = SGINClassification(config, 2)
    # elif config.model == "GAT":
    #     model = GATClassification(config, 2)
       
    model = model.to(device)

    return model

def eval(model, eval_iter, device):
    model.eval()
    eval_loss = 0.0
    for batch_idx, data in enumerate(eval_iter):
        data = data.to(device)
        x = torch.ones((len(data.batch),1)).to(device)
        out = model(data, x)

        loss = model.loss(out, data.y)
        eval_loss += loss.item()

    return eval_loss


def evalacc(model, eval_iter, device):
    model.eval()
    eval_acc = 0.0
    for batch_idx, data in enumerate(eval_iter):
        data = data.to(device)
        x = torch.ones((len(data.batch),1)).to(device)
        acc = model.eval_metric(data, x)
        eval_acc += acc

    return eval_acc / len(eval_iter)

def train_node(model, data, device, config, lr=0.001):
    epochs = config.training_epochs
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        print("Epoch %d" % (e))
        
        model.train()
        epoch_loss = 0.0
        
        optim.zero_grad()
        data = data.to(device)
        out = model(data, data.x)
        loss = model.loss(out[data.train_mask], data.y[data.train_mask])
        epoch_loss = loss.item()
        loss.backward()
        optim.step()
        print("loss: %f" % (epoch_loss))
        
        model.eval()
        eval_loss = 0.0
        out = model(data, data.x)
        eval_loss = model.loss(out[data.test_mask], data.y[data.test_mask])
        
        logits, accs = out, []
        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        print("validation loss: %f" % (eval_loss))
        print("validation acc:",  accs)

def train(model, train_loader, valid_loader, device, config, train_writer, val_writer,
            train_dataset, valid_dataset, lr=0.0001):
    epochs = config.training_epochs
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #optim = torch.optim.SGD(model.parameters(), lr=lr)

    for e in range(epochs):
        print("Epoch %d" % (e))
        # training
        model.train()
        epoch_loss = 0.0
        train_iter = iter(train_loader)
        for batch_idx, data in enumerate(train_iter):
            optim.zero_grad()
            
            data = data.to(device)
            x = torch.ones((len(data.batch),1)).to(device)
            out = model(data, x)

            loss = model.loss(out, data.y)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()

        print("loss: %f" % (epoch_loss))
        train_writer.add_scalar('per_epoch/loss', epoch_loss, e)

#        g = Graph(config, train_dataset[0])
#        fig = g.plot_predictions(model.predictions_to_list( \
#                model.out_to_predictions(model(train_dataset[0].to(device)))))
#        train_writer.add_figure('per_epoch/graph', fig, e)

        # validation
        eval_loss = eval(model, iter(valid_loader), device)
        eval_acc = evalacc(model, iter(valid_loader), device)
        print("validation loss: %f" % (eval_loss))
        print("validation acc: %f" % (eval_acc))
        val_writer.add_scalar('per_epoch/loss', eval_loss, e)

#        g = Graph(config, valid_dataset[0])
#        fig = g.plot_predictions(model.predictions_to_list( \
#                model.out_to_predictions(model(valid_dataset[0].to(device)))))
#        val_writer.add_figure('per_epoch/graph', fig, e)


def main():
    config = Config().parse_args()
    # get device
    device = get_device()

    # load data
    print("loading data...")
    data_dir = '../data'
    train_dataset, valid_dataset, train_loader, valid_loader, test_loader = load_data(config)

    # load model
    print("loading model...")
    model = load_model(device, config)

    summary_dir = os.path.join(config.temp_dir, config.summary_dir)
    if os.path.isdir(summary_dir):
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

    train_writer = SummaryWriter(os.path.join(config.temp_dir, 'summary', 'training'))
    val_writer = SummaryWriter(os.path.join(config.temp_dir, 'summary', 'validation'))

    # train
    print("start training...")
    train(model, train_loader, valid_loader, device, config, train_writer, val_writer,
            train_dataset, valid_dataset)

    #train_node(model, train_dataset, device, config, lr=0.001)

    # predict


if __name__ == "__main__":
    main()

