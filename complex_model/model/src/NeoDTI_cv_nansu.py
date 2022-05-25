# -*- coding: utf-8 -*-
"""
Created on Sun May 08 16:02:48 2022

@author: Shibo Zhou

"""
# import os
# import numpy as np
# import pickle
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import KFold
# from .utils import *
# from torch import nn
# import torch
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split,StratifiedKFold
# import sys
# from optparse import OptionParser
# from .DeepPurpose.utils import *
# from .DeepPurpose import dataset
# from .DeepPurpose import DTI as models
# from . import DataGenerator

from optparse import OptionParser
import DataGenerator
import numpy as np
import torch
import torch.nn.functional as F
from DeepPurpose import dataset
from DeepPurpose.utils import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from torch import nn
from utils import *
import sys

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # features
        self.drug_embedding = weight_variable([num_drug, dim_drug])
        self.protein_embedding = weight_variable([num_protein, dim_protein])
        self.disease_embedding = weight_variable([num_disease, dim_disease])
        self.sideeffect_embedding = weight_variable([num_sideeffect, dim_sideeffect])
        # feature passing weights (maybe different types of nodes can use different weights)
        self.W0 = weight_variable([dim_pass + dim_drug, dim_drug])
        self.b0 = bias_variable([dim_drug])
        # passing 1 times (can be easily extended to multiple passes)
        self.ddn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.dcn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.ddin_die_w = a_layer([num_disease, dim_disease], dim_pass)
        self.dsn_se_w = a_layer([num_sideeffect, dim_sideeffect], dim_pass)
        self.dpn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        self.dpan_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        self.sv_sdn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.pv_ppn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        self.pv_psn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        self.pv_pdin_die_w = a_layer([num_disease, dim_disease], dim_pass)
        self.pv_pdn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.pv_pdan_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.div_didn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        self.div_dipn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        # bi weight
        self.ddr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        self.dcr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        self.ddir = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        self.dsr = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        self.ppr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        self.psr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        self.pdir = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        self.dpr = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)

    def forward(self, drug_drug, drug_drug_normalize,
                drug_chemical, drug_chemical_normalize,
                drug_disease, drug_disease_normalize,
                drug_sideeffect, drug_sideeffect_normalize,
                protein_protein, protein_protein_normalize,
                protein_sequence, protein_sequence_normalize,
                protein_disease, protein_disease_normalize,
                drug_encoder_data, protein_encoder_data,
                disease_drug, disease_drug_normalize,
                disease_protein, disease_protein_normalize,
                sideeffect_drug, sideeffect_drug_normalize,
                drug_protein, drug_protein_normalize,
                protein_drug, protein_drug_normalize,
                drug_protein_mask,
                ):
        l2_loss = cuda(torch.tensor([0], dtype=torch.float64))

        drug_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + \
                       torch.matmul(drug_chemical_normalize, a_cul(self.drug_embedding, *self.dcn_de_w)) + \
                       torch.matmul(drug_disease_normalize, a_cul(self.disease_embedding, *self.ddin_die_w)) + \
                       torch.matmul(drug_sideeffect_normalize, a_cul(self.sideeffect_embedding, *self.dsn_se_w)) + \
                       torch.matmul(drug_protein_normalize,
                                    a_cul(self.protein_embedding, *self.dpn_pe_w)) + drug_encoder_data, \
                       self.drug_embedding], axis=1), self.W0) + self.b0), dim=1)

        sideeffect_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(sideeffect_drug_normalize, a_cul(self.drug_embedding, *self.sv_sdn_de_w)), \
                       self.sideeffect_embedding], axis=1), self.W0) + self.b0), dim=1)

        protein_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(protein_protein_normalize, a_cul(self.protein_embedding, *self.pv_ppn_pe_w)) + \
                       torch.matmul(protein_sequence_normalize, a_cul(self.protein_embedding, *self.pv_psn_pe_w)) + \
                       torch.matmul(protein_disease_normalize, a_cul(self.disease_embedding, *self.pv_pdin_die_w)) + \
                       torch.matmul(protein_drug_normalize,
                                    a_cul(self.drug_embedding, *self.pv_pdn_de_w)) + protein_encoder_data, \
                       self.protein_embedding], axis=1), self.W0) + self.b0), dim=1)

        disease_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(disease_drug_normalize, a_cul(self.drug_embedding, *self.div_didn_de_w)) + \
                       torch.matmul(disease_protein_normalize, a_cul(self.protein_embedding, *self.div_dipn_pe_w)), \
                       self.disease_embedding], axis=1), self.W0) + self.b0), dim=1)

        drug_representation = drug_vector1
        protein_representation = protein_vector1
        disease_representation = disease_vector1
        sideeffect_representation = sideeffect_vector1

        # reconstructing networks
        drug_drug_reconstruct = bi_cul(drug_representation, drug_representation, *self.ddr)
        drug_drug_reconstruct_loss = torch.sum(
            torch.multiply((drug_drug_reconstruct - drug_drug), (drug_drug_reconstruct - drug_drug)))

        drug_chemical_reconstruct = bi_cul(drug_representation, drug_representation, *self.dcr)
        drug_chemical_reconstruct_loss = torch.sum(
            torch.multiply((drug_chemical_reconstruct - drug_chemical), (drug_chemical_reconstruct - drug_chemical)))

        drug_disease_reconstruct = bi_cul(drug_representation, disease_representation, *self.ddir)
        drug_disease_reconstruct_loss = torch.sum(
            torch.multiply((drug_disease_reconstruct - drug_disease), (drug_disease_reconstruct - drug_disease)))

        drug_sideeffect_reconstruct = bi_cul(drug_representation, sideeffect_representation, *self.dsr)
        drug_sideeffect_reconstruct_loss = torch.sum(torch.multiply((drug_sideeffect_reconstruct - drug_sideeffect),
                                                                    (drug_sideeffect_reconstruct - drug_sideeffect)))

        protein_protein_reconstruct = bi_cul(protein_representation, protein_representation, *self.ppr)
        protein_protein_reconstruct_loss = torch.sum(torch.multiply((protein_protein_reconstruct - protein_protein),
                                                                    (protein_protein_reconstruct - protein_protein)))

        protein_sequence_reconstruct = bi_cul(protein_representation, protein_representation, *self.psr)
        protein_sequence_reconstruct_loss = torch.sum(torch.multiply((protein_sequence_reconstruct - protein_sequence),
                                                                     (protein_sequence_reconstruct - protein_sequence)))

        protein_disease_reconstruct = bi_cul(protein_representation, disease_representation, *self.pdir)
        protein_disease_reconstruct_loss = torch.sum(torch.multiply((protein_disease_reconstruct - protein_disease),
                                                                    (protein_disease_reconstruct - protein_disease)))

        drug_protein_reconstruct = bi_cul(drug_representation, protein_representation, *self.dpr)
        tmp = torch.multiply(drug_protein_mask, (drug_protein_reconstruct - drug_protein))
        drug_protein_reconstruct_loss = torch.sum(torch.multiply(tmp, tmp))  # / (torch.sum(self.drug_protein_mask)

        for param in model.parameters():
            l2_loss += torch.norm(param, 2)

        loss = drug_protein_reconstruct_loss + 1.0 * (drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss +
                                                      drug_disease_reconstruct_loss + drug_sideeffect_reconstruct_loss +
                                                      protein_protein_reconstruct_loss + protein_sequence_reconstruct_loss +
                                                      protein_disease_reconstruct_loss) + l2_loss
        return loss, drug_protein_reconstruct_loss, drug_protein_reconstruct


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def train_and_evaluate_(DTItrain, DTItest, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    model.zero_grad()

    for i in range(num_steps):
        print('doing round %s step %s / %s' % (round, i, num_steps))
        model.train()
        mask_ = cuda(torch.tensor(mask))
        protein_drug_normalize_ = cuda(torch.tensor(protein_drug_normalize))
        protein_drug_ = cuda(torch.tensor(protein_drug))
        drug_protein_normalize_ = cuda(torch.tensor(drug_protein_normalize))
        drug_protein_ = cuda(torch.tensor(drug_protein))
        global sideeffect_drug_normalize
        sideeffect_drug_normalize_ = cuda(torch.tensor(sideeffect_drug_normalize))
        global sideeffect_drug
        sideeffect_drug_ = cuda(torch.tensor(sideeffect_drug))
        global disease_protein_normalize
        disease_protein_normalize_ = cuda(torch.tensor(disease_protein_normalize))
        global disease_protein
        disease_protein_ = cuda(torch.tensor(disease_protein))
        global disease_drug_normalize
        disease_drug_normalize_ = cuda(torch.tensor(disease_drug_normalize))
        global disease_drug
        disease_drug_ = cuda(torch.tensor(disease_drug))
        global protein_disease_normalize
        protein_disease_normalize_ = cuda(torch.tensor(protein_disease_normalize))
        global protein_disease
        protein_disease_ = cuda(torch.tensor(protein_disease))
        global protein_sequence_normalize
        protein_sequence_normalize_ = cuda(torch.tensor(protein_sequence_normalize))
        global protein_sequence
        protein_sequence_ = cuda(torch.tensor(protein_sequence))
        global protein_protein_normalize
        protein_protein_normalize_ = cuda(torch.tensor(protein_protein_normalize))
        global protein_protein
        protein_protein_ = cuda(torch.tensor(protein_protein))
        global drug_sideeffect_normalize
        drug_sideeffect_normalize_ = cuda(torch.tensor(drug_sideeffect_normalize))
        global drug_sideeffect
        drug_sideeffect_ = cuda(torch.tensor(drug_sideeffect))
        global drug_disease_normalize
        drug_disease_normalize_ = cuda(torch.tensor(drug_disease_normalize))
        global drug_disease
        drug_disease_ = cuda(torch.tensor(drug_disease))
        global drug_chemical_normalize
        drug_chemical_normalize_ = cuda(torch.tensor(drug_chemical_normalize))
        global drug_chemical
        drug_chemical_ = cuda(torch.tensor(drug_chemical))
        global drug_drug_normalize
        drug_drug_normalize_ = cuda(torch.tensor(drug_drug_normalize))
        global drug_drug
        drug_drug_ = cuda(torch.tensor(drug_drug))
        global drug_encoder_data
        drug_encoder_data_ = cuda(drug_encoder_data)
        global protein_encoder_data
        protein_encoder_data_ = cuda(protein_encoder_data)

        tloss, dtiloss, results = model(
            drug_drug_, drug_drug_normalize_, \
            drug_chemical_, drug_chemical_normalize_, \
            drug_disease_, drug_disease_normalize_, \
            drug_sideeffect_, drug_sideeffect_normalize_, \
            protein_protein_, protein_protein_normalize_, \
            protein_sequence_, protein_sequence_normalize_, \
            protein_disease_, protein_disease_normalize_, \
            drug_encoder_data_, protein_encoder_data_, \
            disease_drug_, disease_drug_normalize_, \
            disease_protein_, disease_protein_normalize_, \
            sideeffect_drug_, sideeffect_drug_normalize_, \
            drug_protein_, drug_protein_normalize_, \
            protein_drug_, protein_drug_normalize_, \
            mask_)
        tloss.backward(retain_graph=True)
        # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
    pred_list = []
    ground_truth = []
    ii = 1
    for ele in DTItest:
        # print('Testing %s / %s' % (ii, len(DTItest)))
        pred_list.append(results[ele[0], ele[1]].cpu().detach().numpy())
        ground_truth.append(ele[2])
        # pred_list.append(results[ele[0], ele[1]])
        print(results[ele[0], ele[1]], "-> ", ele[2])
        ii += 1
    test_auc = roc_auc_score(ground_truth, pred_list)
    test_aupr = average_precision_score(ground_truth, pred_list)

    print('test_auc: ',test_auc)
    print('test_aupr: ', test_aupr)
    return test_auc, test_aupr


def train_and_evaluate(DTItrain, DTIvalid, DTItest, round, verbose=True, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    model.zero_grad()

    for i in range(num_steps):
        print('doing round %s step %s / %s' % (round, i, num_steps))
        model.train()
        mask_ = cuda(torch.tensor(mask))
        protein_drug_normalize_ = cuda(torch.tensor(protein_drug_normalize))
        protein_drug_ = cuda(torch.tensor(protein_drug))
        drug_protein_normalize_ = cuda(torch.tensor(drug_protein_normalize))
        drug_protein_ = cuda(torch.tensor(drug_protein))
        global sideeffect_drug_normalize
        sideeffect_drug_normalize_ = cuda(torch.tensor(sideeffect_drug_normalize))
        global sideeffect_drug
        sideeffect_drug_ = cuda(torch.tensor(sideeffect_drug))
        global disease_protein_normalize
        disease_protein_normalize_ = cuda(torch.tensor(disease_protein_normalize))
        global disease_protein
        disease_protein_ = cuda(torch.tensor(disease_protein))
        global disease_drug_normalize
        disease_drug_normalize_ = cuda(torch.tensor(disease_drug_normalize))
        global disease_drug
        disease_drug_ = cuda(torch.tensor(disease_drug))
        global protein_disease_normalize
        protein_disease_normalize_ = cuda(torch.tensor(protein_disease_normalize))
        global protein_disease
        protein_disease_ = cuda(torch.tensor(protein_disease))
        global protein_sequence_normalize
        protein_sequence_normalize_ = cuda(torch.tensor(protein_sequence_normalize))
        global protein_sequence
        protein_sequence_ = cuda(torch.tensor(protein_sequence))
        global protein_protein_normalize
        protein_protein_normalize_ = cuda(torch.tensor(protein_protein_normalize))
        global protein_protein
        protein_protein_ = cuda(torch.tensor(protein_protein))
        global drug_sideeffect_normalize
        drug_sideeffect_normalize_ = cuda(torch.tensor(drug_sideeffect_normalize))
        global drug_sideeffect
        drug_sideeffect_ = cuda(torch.tensor(drug_sideeffect))
        global drug_disease_normalize
        drug_disease_normalize_ = cuda(torch.tensor(drug_disease_normalize))
        global drug_disease
        drug_disease_ = cuda(torch.tensor(drug_disease))
        global drug_chemical_normalize
        drug_chemical_normalize_ = cuda(torch.tensor(drug_chemical_normalize))
        global drug_chemical
        drug_chemical_ = cuda(torch.tensor(drug_chemical))
        global drug_drug_normalize
        drug_drug_normalize_ = cuda(torch.tensor(drug_drug_normalize))
        global drug_drug
        drug_drug_ = cuda(torch.tensor(drug_drug))
        global drug_encoder_data
        drug_encoder_data_ = cuda(drug_encoder_data)
        global protein_encoder_data
        protein_encoder_data_ = cuda(protein_encoder_data)

        tloss, dtiloss, results = model(
            drug_drug_, drug_drug_normalize_, \
            drug_chemical_, drug_chemical_normalize_, \
            drug_disease_, drug_disease_normalize_, \
            drug_sideeffect_, drug_sideeffect_normalize_, \
            protein_protein_, protein_protein_normalize_, \
            protein_sequence_, protein_sequence_normalize_, \
            protein_disease_, protein_disease_normalize_, \
            drug_encoder_data_, protein_encoder_data_, \
            disease_drug_, disease_drug_normalize_, \
            disease_protein_, disease_protein_normalize_, \
            sideeffect_drug_, sideeffect_drug_normalize_, \
            drug_protein_, drug_protein_normalize_, \
            protein_drug_, protein_drug_normalize_, \
            mask_)
        tloss.backward(retain_graph=True)
        # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
        if i % 25 == 0 and verbose == True:
            # if i == 0:
            print('step', i, 'total and dtiloss', tloss, dtiloss)

            pred_list = []
            ground_truth = []
            ii = 1
            for ele in DTIvalid:
                print('Evaluating %s / %s' % (ii, len(DTIvalid)))
                pred_list.append(results[ele[0], ele[1]].cpu().detach().numpy())
                ground_truth.append(ele[2])
                ii += 1
            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)
            if valid_aupr >= best_valid_aupr:
                torch.save(model, 'model_round_%s.pkl' % (round))
                best_valid_aupr = valid_aupr
                best_valid_auc = valid_auc
                pred_list = []
                ground_truth = []
                ii = 1
                for ele in DTItest:
                    print('Testing %s / %s' % (ii, len(DTItest)))
                    pred_list.append(results[ele[0], ele[1]].cpu().detach().numpy())
                    ground_truth.append(ele[2])
                    ii += 1
                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
            print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)

    return best_valid_auc, best_valid_aupr, test_auc, test_aupr


parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k")
parser.add_option("-t", "--t", default="o", help="Test scenario")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio")

(opts, args) = parser.parse_args()

# network encoder data

# network_path = '/infodev1/home/mid/project/network/benchmark/data/datasets/'
network_path = '/infodev1/non-phi-data/nansu/benchmark_2.0'
opts_d=1024
opts_k=256
opts_n=2

# result_file = dir + '/NeoDTI_results_phase2_append.txt'
result_file = network_path + '/newmodel_NeoDTI_test.txt'


if os.path.exists(result_file):
    os.remove(result_file)



# drug_idx_file = network_path + 'DTINet_revision/drug.idx'
# target_idx_file = network_path + 'DTINet_revision/target.idx'
# smile_file = network_path + 'experiment/smile.txt'
# sequence_file = network_path + 'experiment/sequence.txt'

drug_idx_file = network_path + '/drug.idx'
target_idx_file = network_path + '/target.idx'

smile_file = network_path + '/smile.txt'
sequence_file = network_path + '/sequence.txt'

drugs = DataGenerator.readIdx(drug_idx_file)
targets = DataGenerator.readIdx(target_idx_file)

print('len(drugs): ', len(drugs))
print('len(targets): ', len(targets))

X_drugs = dataset.read_drugfile(path=smile_file, drugs=drugs)
X_targets = dataset.read_targetfile(path=sequence_file, targets=targets)

print('X_drugs.shape: ', X_drugs.shape)
print('X_targets.shape: ', X_targets.shape)
print(X_drugs)
print(X_targets)

drug_encoding, target_encoding = 'CNN', 'CNN'

df_data_drug, df_data_target = data_process_(X_drug=X_drugs, X_target=X_targets, drug_encoding=drug_encoding,
                                             target_encoding=target_encoding)

config = generate_config(drug_encoding=drug_encoding,
                         target_encoding=target_encoding,
                         drug_batch_size=708,
                         protein_batch_size=1512,
                         hidden_dim_drug=1024,
                         hidden_dim_protein=1024,
                         )

drug_encoder_data, protein_encoder_data = data_encoder_(df_data_drug, df_data_target, config)
print('drug_encoder_data: ', drug_encoder_data.shape)
print('protein_encoder_data: ', protein_encoder_data.shape)

print('New data encoder finish')

# load network

# drug_drug_file = network_path + 'DTINet_revision/mat_drug_drug.txt'
# drug_chemical_file = network_path + 'DTINet_revision/Similarity_Matrix_Drugs.txt'
# drug_disease_file = network_path + 'DTINet_revision/mat_drug_disease.txt'
# drug_sideeffect_file = network_path + 'DTINet_revision/mat_drug_sider.txt'
# protein_protein_file = network_path + 'DTINet_revision/mat_target_target.txt'
# protein_sequence_file = network_path + 'DTINet_revision/Similarity_Matrix_Targets.txt'
# protein_disease_file = network_path + 'DTINet_revision/mat_target_disease.txt'

drug_drug_file = network_path + '/mat_drug_drug.txt'
drug_chemical_file = network_path + '/Similarity_Matrix_Drugs.txt'
drug_disease_file = network_path + '/mat_drug_disease.txt'
drug_sideeffect_file = network_path + '/mat_drug_sider.txt'
protein_protein_file = network_path + '/mat_target_target.txt'
protein_sequence_file = network_path + '/Similarity_Matrix_Targets.txt'
protein_disease_file = network_path + '/mat_target_disease.txt'

drug_drug = np.loadtxt(drug_drug_file)
drug_chemical = np.loadtxt(drug_chemical_file)
drug_disease = np.loadtxt(drug_disease_file)
drug_sideeffect = np.loadtxt(drug_sideeffect_file)
protein_protein = np.loadtxt(protein_protein_file)
protein_sequence = np.loadtxt(protein_sequence_file)
protein_disease = np.loadtxt(protein_disease_file)

disease_drug = drug_disease.T
sideeffect_drug = drug_sideeffect.T
disease_protein = protein_disease.T

drug_drug_normalize = row_normalize(drug_drug, True)
drug_chemical_normalize = row_normalize(drug_chemical, True)
drug_disease_normalize = row_normalize(drug_disease, False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect, False)
protein_protein_normalize = row_normalize(protein_protein, True)
protein_sequence_normalize = row_normalize(protein_sequence, True)
protein_disease_normalize = row_normalize(protein_disease, False)
disease_drug_normalize = row_normalize(disease_drug, False)
disease_protein_normalize = row_normalize(disease_protein, False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug, False)

num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)
dim_drug = int(opts_d)
dim_protein = int(opts_d)
dim_disease = int(opts_d)
dim_sideeffect = int(opts_d)
dim_pred = int(opts_k)
dim_pass = int(opts_d)

print('dim_drug', dim_drug)
print('dim_protein', dim_protein)
print('dim_disease', dim_disease)
print('dim_sideeffect', dim_sideeffect)
print('dim_pred', dim_pred)
print('dim_pass', dim_pass)
print('num_drug: ', num_drug)
print('num_protein: ', num_protein)
print('num_disease: ', num_disease)
print('num_sideeffect: ', num_sideeffect)
print('drug_drug_normalize: ', drug_drug_normalize.shape)
print('drug_chemical_normalize: ', drug_chemical_normalize.shape)
print('drug_disease_normalize: ', drug_disease_normalize.shape)
print('drug_sideeffect_normalize: ', drug_sideeffect_normalize.shape)
print('protein_protein_normalize: ', protein_protein_normalize.shape)
print('protein_sequence_normalize: ', protein_sequence_normalize.shape)
print('protein_disease_normalize: ', protein_disease_normalize.shape)
print('disease_drug_normalize: ', disease_drug_normalize.shape)
print('disease_protein_normalize: ', disease_protein_normalize.shape)
print('sideeffect_drug_normalize: ', sideeffect_drug_normalize.shape)

lr = 0.001
model = cuda(Model())
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train_file = network_path + 'experiment/internal/train_0_general.nt'
#
# test_file = network_path + 'experiment/internal/test_0_general.nt'

train_file = network_path + '/internal/train_0_general.nt'

test_file = network_path + '/internal/test_0_general.nt'

DTItrain = DataGenerator.readTrain(train_file, drug_idx_file, target_idx_file)

DTItest = DataGenerator.readTest(test_file, drug_idx_file, target_idx_file)

train_and_evaluate_(DTItrain=DTItrain, DTItest=DTItest, num_steps=10)
