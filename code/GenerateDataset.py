# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:58:54 2018

@author: Michael
"""

import os
import numpy as np
import pandas as pd
import time
import random
import pickle
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import math
import torch
from torch_geometric.data import Data
import scipy
import torch.nn.functional as F
from networkx.convert_matrix import to_scipy_sparse_matrix, to_numpy_array
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

for m in range(1,3):
    i=0
    while i<500:
        G1=nx.fast_gnp_random_graph(100,0.07)
        S1=to_numpy_array(G1)
        T1=torch.from_numpy(S1)
        TS1=to_sparse(T1)
        connect=nx.edge_connectivity(G1)
        if connect==m:
            data=Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*connect)
            torch.save(data,"./.data/connected/connect_" + str(m) + "_" + str(i) + ".pt")
            i+=1
for m in range(3,5):
    i=0
    while i<500:
        G1=nx.fast_gnp_random_graph(100,0.07)
        S1=to_numpy_array(G1)
        T1=torch.from_numpy(S1)
        TS1=to_sparse(T1)
        connect=nx.edge_connectivity(G1)
        clique=nx.graph_clique_number(G1)
        if connect>0 and clique==m:
            data=Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*m)
            torch.save(data,"./.data/clique/clique_" + str(m) + "_" + str(i) + ".pt")
            i+=1