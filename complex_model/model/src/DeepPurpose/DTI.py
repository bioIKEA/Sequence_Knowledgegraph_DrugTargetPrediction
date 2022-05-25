import os
import copy
import torch
import torch.nn.functional as F
from torch import nn 
from DeepPurpose.encoders import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
torch.manual_seed(2)
np.random.seed(3)



class Classifier_drug(nn.Sequential):
    def __init__(self, model_drug, **config):
        super(Classifier_drug, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']

        self.model_drug = model_drug

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug] + self.hidden_dims + [1]
        
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v_D):
        # each encoding
        v_D = self.model_drug(v_D)   # 128x128
        return v_D

class Classifier_protein(nn.Sequential):
    def __init__(self, model_protein, **config):
        super(Classifier_protein, self).__init__()
        self.input_dim_protein = config['hidden_dim_protein']

        self.model_protein = model_protein

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_protein] + self.hidden_dims + [1]
        
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v_P):
        # each encoding
        v_P = self.model_protein(v_P) # 128x256
        return v_P


def model_initialize_drug(**config):
    model_drug = CNN('drug', **config)
    model = Classifier_drug(model_drug, **config)
    return model
    
def model_initialize_protein(**config):
    model_protein = CNN('protein', **config)
    model = Classifier_protein(model_protein, **config)
    return model

