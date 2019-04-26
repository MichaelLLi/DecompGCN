import torch
from torch_geometric.data import DataLoader

import os
from tensorboardX import SummaryWriter
import shutil

#from Graph import Graph
from GraphDataset import RandomConnectedGraph, RandomCliqueGraph, RandomTreeCycleGraph
from config import Config
from GCNClassification import GCNClassification, GINClassification, \
        SGConvClassification, SGINClassification, GATClassification
from random import shuffle
from torch_geometric.datasets import TUDataset

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(config):
    print("Task: ", config.task)
    if config.task=="connectivity":
        dataset = RandomConnectedGraph()
    elif config.task=="clique":
        dataset = RandomCliqueGraph()
    elif config.task=="tree_cycle":
        dataset = RandomTreeCycleGraph()
    elif config.task=="reddit-b":
        dataset = TUDataset(root="/tmp/redditb",name="REDDIT-BINARY")
    if config.task!="reddit-b":
        data_list=[]
        for i in range(dataset.__len__()):
            data_list.append(dataset[i])
    else:
        data_list=dataset
    
    train_idx = int(len(data_list) * (1 - config.test_split -
                                            config.validation_split))
    valid_idx = int(len(data_list) * (1 - config.test_split))
    if config.task!="reddit-b":
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

def load_model(device, config):
    print("Model: ", config.model)
    if config.task in ['clique','connectivity','reddit-b', 'tree_cycle']:
        if config.model == "GIN":
            model = GINClassification(config, 2)
        elif config.model == "GCN":
            model = GCNClassification(config,2)
        elif config.model == "SGConv":
            model = SGConvClassification(config, 2)
        elif config.model == "SGIN":
            model = SGINClassification(config, 2)
        elif config.model == "GAT":
            model = GATClassification(config, 2)
    model = model.to(device)

    return model

def eval(model, eval_iter, device):
    model.eval()
    eval_loss = 0.0
    for batch_idx, data in enumerate(eval_iter):
        data = data.to(device)
        out = model(data)

        loss = model.loss(out, data.y)
        eval_loss += loss.item()

    return eval_loss

def evalacc(model, eval_iter, device):
    model.eval()
    eval_acc = 0.0
    for batch_idx, data in enumerate(eval_iter):
        data = data.to(device)
        acc = model.eval_metric(data)
        eval_acc += acc

    return eval_acc / len(eval_iter)

def train(model, train_loader, valid_loader, device, config, train_writer, val_writer,
            train_dataset, valid_dataset, epochs=300, lr=0.001):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        print("Epoch %d" % (e))
        # training
        model.train()
        epoch_loss = 0.0
        train_iter = iter(train_loader)
        for batch_idx, data in enumerate(train_iter):
            optim.zero_grad()
            
            data = data.to(device)
            out = model(data)

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

    # predict


if __name__ == "__main__":
    main()

