import pandas as pd
import numpy as np
import wget
from zipfile import ZipFile
import json
import os

'''
Acknowledgement:

The Davis Dataset can be found in http://staff.cs.utu.fi/~aatapa/data/DrugTarget/.

'''

def convert_y_unit(y, from_, to_):
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)    
    # basis as nM
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10**(-y) / 1e-9

    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y*1e-9)
    elif to_ == 'nM':
        y = y
        
    if array_flag:
        return y[0]
    return y

def read_drugfile(path, drugs):
    # a line in the file is SMILES
    try:
        file = open(path, "r")
    except:
        print('Path Not Found, please double check!')
    X_drug = [None]*len(drugs)
    drug_value={}

    for aline in file:
        values = aline.split()
        key=values[0]
        value = values[1]
        drug_value[key]=value
    file.close()

    for i in range(len(drugs)):
        if drugs[i] in drug_value:
            X_drug[i]=drug_value.get(drugs[i])
        else:
            X_drug[i] = 'CCCCCCCCCCCCCCCCCCCC'

    return np.array(X_drug)


def read_targetfile(path, targets):
    # a line in the file is SMILES
    try:
        file = open(path, "r")
    except:
        print('Path Not Found, please double check!')
    X_target = [None]*len(targets)
    target_value = {}

    for aline in file:
        values = aline.split()
        key = values[0]
        value = values[1]
        target_value[key]=value
    file.close()

    for i in range(len(targets)):
        if targets[i] in target_value:
            X_target[i] = target_value.get(targets[i])
        else:
            X_target[i] = 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'

    return np.array(X_target)


def read_file_training_dataset_drug_target_pairs(path):
    # a line in the file is SMILES Target_seq score
    try:
        file = open(path, "r")
    except:
        print('Path Not Found, please double check!')
    X_drug = []
    X_target = []
    y = []
    for aline in file:
        values = aline.split()
        X_drug.append(values[0])
        X_target.append(values[1])
        y.append(float(values[2]))
    file.close()
    return np.array(X_drug), np.array(X_target), np.array(y)

def load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30):
    print('Beginning Processing...')

    if not os.path.exists(path):
        os.makedirs(path)

    affinity = pd.read_csv(path + '/DAVIS/affinity.txt', header=None, sep = ' ')

    with open(path + '/DAVIS/target_seq.txt') as f:
        target = json.load(f)

    with open(path + '/DAVIS/SMILES.txt') as f:
        drug = json.load(f)
        #print(drug)
    target = list(target.values())
    drug = list(drug.values())

    SMILES = []
    Target_seq = []
    y = []

    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])

    if binary:
        print('Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        if convert_to_log:
            print('Default set to logspace (nM -> p) for easier regression')
            y = convert_y_unit(np.array(y), 'nM', 'p')
        else:
            y = y
    print('Done!')
    return np.array(SMILES), np.array(Target_seq), np.array(y)
