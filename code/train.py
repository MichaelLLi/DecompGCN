import torch
from torch_geometric.data import DataLoader

import os
from tensorboardX import SummaryWriter
import shutil

from Graph import Graph
from GraphDataset import RandomConnectedGraph, RandomCliqueGraph
from config import Config
from GCNClassification import GCNClassification
from random import shuffle

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(config):
    if config.task=="connectivity":
        dataset = RandomConnectedGraph()
    elif config.task=="clique":
        dataset = RandomCliqueGraph()
    data_list=[]
    for i in range(dataset.__len__()):
        data_list.append(dataset[i])
    train_idx = int(config.num_graphs * (1 - config.test_split - 
                                            config.validation_split))
    valid_idx = int(config.num_graphs * (1 - config.test_split))
    shuffle(data_list)
    train_dataset = data_list[:train_idx]
    valid_dataset = data_list[train_idx:valid_idx]
    test_dataset = data_list[valid_idx:]

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size_train, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size_train, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size_eval, shuffle=True)

    return train_dataset, valid_dataset, iter(train_loader), iter(valid_loader), iter(test_loader)

def load_model(device, config):
    if config.task == 'clique':
        model = GCNClassification(config, 2)
    elif config.task == 'connectivity':
        model = GCNClassification(config, 2)
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


def train(model, train_iter, valid_iter, device, config, train_writer, val_writer, 
            train_dataset, valid_dataset, epochs=10, lr=0.1):
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for e in range(epochs):
        print("Epoch %d" % (e))
        # training
        model.train()
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_iter):
            optim.zero_grad()

            data = data.to(device)
            out = model(data)

            loss = model.loss(out, data.y)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()

        train_writer.add_scalar('per_epoch/loss', epoch_loss, e)
        
#        g = Graph(config, train_dataset[0])
#        fig = g.plot_predictions(model.predictions_to_list( \
#                model.out_to_predictions(model(train_dataset[0].to(device)))))
#        train_writer.add_figure('per_epoch/graph', fig, e)

        # validation
        eval(model, valid_iter, device)

        val_writer.add_scalar('per_epoch/loss', epoch_loss, e)
        
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
    train_dataset, valid_dataset, train_iter, valid_iter, test_iter = load_data(config)

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
    train(model, train_iter, valid_iter, device, config, train_writer, val_writer,
            train_dataset, valid_dataset)

    # predict


if __name__ == "__main__":
    main()

