#import tensorflow as tf
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def weight_variable(shape):
    variable = nn.Parameter(cuda(torch.Tensor(*shape).double()))
    return nn.init.normal_(variable, std = 0.1)

def bias_variable(shape):
    variable = nn.Parameter(cuda(torch.Tensor(*shape).double()))
    return nn.init.constant_(variable, val = 0.1)

def a_layer(x_shape,units):
    W = weight_variable([x_shape[1], units])
    b = bias_variable([units])
    return W, b

def a_cul(x, w, b):
    return F.relu(torch.matmul(x, w) + b).double()
    
def bi_cul(x0, x1, w0, w1):
    return torch.matmul(torch.matmul(x0, w0), 
                            torch.transpose(torch.matmul(x1, w1), 0, 1)).double()

def bi_layer(x0_dim_1,x1_dim_1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0_dim_1,dim_pred])
        W1p = weight_variable([x1_dim_1,dim_pred])
        return W0p, W1p
    else:
        W0p = weight_variable([x0_dim_1,dim_pred])
        return W0p, W0p

               