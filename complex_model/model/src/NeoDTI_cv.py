# -*- coding: utf-8 -*-
"""
Created on Sun May 08 16:02:48 2022

@author: Shibo Zhou

"""
import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from utils import *
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split,StratifiedKFold
import sys
from optparse import OptionParser
from DeepPurpose.utils import *
from DeepPurpose import dataset
from DeepPurpose import DTI as models



parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n","--n",default=1, help="global norm to be clipped")
parser.add_option("-k","--k",default=512,help="The dimension of project matrices k")
parser.add_option("-t","--t",default = "o",help="Test scenario")
parser.add_option("-r","--r",default = "ten",help="positive negative ratio")

(opts, args) = parser.parse_args()



def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol) 

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)   
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12  
    new_matrix = a_matrix / row_sums[:, np.newaxis] 
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0 
    return new_matrix


# network encoder data
X_drugs, X_targets, y = dataset.load_process_DAVIS(path = '../data', binary = False, convert_to_log = True, threshold = 30)


drug_encoding, target_encoding = 'CNN', 'CNN'

train, val, test = data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)


config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,                        
                         drug_batch_size = 708, 
                         protein_batch_size = 1512,
                         hidden_dim_drug = 1024, 
                         hidden_dim_protein = 1024, 
                        )


drug_encoder_data, protein_encoder_data = data_encoder(train, config)
print('New data encoder finish')

# load network
network_path = '../data/'

drug_drug = np.loadtxt(network_path+'mat_drug_drug.txt')

true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')
drug_chemical=drug_chemical[:true_drug,:true_drug]

drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt') 
drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')

disease_drug = drug_disease.T 
sideeffect_drug = drug_sideeffect.T

protein_protein = np.loadtxt(network_path+'mat_protein_protein.txt') 
protein_sequence = np.loadtxt(network_path+'Similarity_Matrix_Proteins.txt') 
protein_disease = np.loadtxt(network_path+'mat_protein_disease.txt') 

disease_protein = protein_disease.T

#normalize network for mean pooling aggregation
drug_drug_normalize = row_normalize(drug_drug,True)
drug_chemical_normalize = row_normalize(drug_chemical,True)
drug_disease_normalize = row_normalize(drug_disease,False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect,False)

protein_protein_normalize = row_normalize(protein_protein,True)
protein_sequence_normalize = row_normalize(protein_sequence,True)
protein_disease_normalize = row_normalize(protein_disease,False)

disease_drug_normalize = row_normalize(disease_drug,False)
disease_protein_normalize = row_normalize(disease_protein,False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug,False)



#define computation graph
num_drug = len(drug_drug_normalize) 
num_protein = len(protein_protein_normalize)   
num_disease = len(disease_protein_normalize)  
num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d) 
dim_protein = int(opts.d)  
dim_disease = int(opts.d)
dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)       
dim_pass = int(opts.d) 



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #features
        self.drug_embedding = weight_variable([num_drug,dim_drug])           
        self.protein_embedding = weight_variable([num_protein,dim_protein])  
        self.disease_embedding = weight_variable([num_disease,dim_disease]) 
        self.sideeffect_embedding = weight_variable([num_sideeffect,dim_sideeffect])  
        #feature passing weights (maybe different types of nodes can use different weights)
        self.W0 = weight_variable([dim_pass+dim_drug, dim_drug])              
        self.b0 = bias_variable([dim_drug])                                    
        #passing 1 times (can be easily extended to multiple passes)
        self.ddn_de_w = a_layer([num_drug,dim_drug], dim_pass)                
        self.dcn_de_w = a_layer([num_drug,dim_drug], dim_pass)
        self.ddin_die_w = a_layer([num_disease,dim_disease], dim_pass)
        self.dsn_se_w = a_layer([num_sideeffect,dim_sideeffect], dim_pass)
        self.dpn_pe_w = a_layer([num_protein,dim_protein], dim_pass)
        self.dpan_pe_w = a_layer([num_protein,dim_protein], dim_pass)
        self.sv_sdn_de_w = a_layer([num_drug,dim_drug], dim_pass)
        self.pv_ppn_pe_w = a_layer([num_protein,dim_protein], dim_pass)
        self.pv_psn_pe_w = a_layer([num_protein,dim_protein], dim_pass)
        self.pv_pdin_die_w = a_layer([num_disease,dim_disease], dim_pass)
        self.pv_pdn_de_w = a_layer([num_drug,dim_drug], dim_pass)
        self.pv_pdan_de_w = a_layer([num_drug,dim_drug], dim_pass)
        self.div_didn_de_w = a_layer([num_drug,dim_drug], dim_pass)
        self.div_dipn_pe_w = a_layer([num_protein,dim_protein], dim_pass)
        #bi weight
        self.ddr = bi_layer(dim_drug,dim_drug, sym=True, dim_pred=dim_pred)   
        self.dcr = bi_layer(dim_drug,dim_drug, sym=True, dim_pred=dim_pred)
        self.ddir = bi_layer(dim_drug,dim_drug, sym=False, dim_pred=dim_pred)
        self.dsr = bi_layer(dim_drug,dim_drug, sym=False, dim_pred=dim_pred)
        self.ppr = bi_layer(dim_drug,dim_drug, sym=True, dim_pred=dim_pred)
        self.psr = bi_layer(dim_drug,dim_drug, sym=True, dim_pred=dim_pred)
        self.pdir = bi_layer(dim_drug,dim_drug, sym=False, dim_pred=dim_pred)
        self.dpr = bi_layer(dim_drug,dim_drug, sym=False, dim_pred=dim_pred)
        
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
        
        
        l2_loss = cuda(torch.tensor([0], dtype = torch.float64))
        
        drug_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
                torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + \
                torch.matmul(drug_chemical_normalize, a_cul(self.drug_embedding, *self.dcn_de_w)) + \
                torch.matmul(drug_disease_normalize, a_cul(self.disease_embedding, *self.ddin_die_w)) + \
                torch.matmul(drug_sideeffect_normalize, a_cul(self.sideeffect_embedding, *self.dsn_se_w)) + \
                torch.matmul(drug_protein_normalize, a_cul(self.protein_embedding, *self.dpn_pe_w)) + drug_encoder_data, \
                self.drug_embedding], axis=1), self.W0)+self.b0),dim=1)

        sideeffect_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
                torch.cat([torch.matmul(sideeffect_drug_normalize, a_cul(self.drug_embedding, *self.sv_sdn_de_w)), \
                self.sideeffect_embedding], axis=1), self.W0)+self.b0),dim=1)

        protein_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
                torch.cat([torch.matmul(protein_protein_normalize, a_cul(self.protein_embedding, *self.pv_ppn_pe_w)) + \
                torch.matmul(protein_sequence_normalize, a_cul(self.protein_embedding, *self.pv_psn_pe_w)) + \
                torch.matmul(protein_disease_normalize, a_cul(self.disease_embedding, *self.pv_pdin_die_w)) + \
                torch.matmul(protein_drug_normalize, a_cul(self.drug_embedding, *self.pv_pdn_de_w)) + protein_encoder_data, \
                self.protein_embedding], axis=1), self.W0)+self.b0),dim=1)

        disease_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
                torch.cat([torch.matmul(disease_drug_normalize, a_cul(self.drug_embedding, *self.div_didn_de_w)) + \
                torch.matmul(disease_protein_normalize, a_cul(self.protein_embedding, *self.div_dipn_pe_w)), \
                self.disease_embedding], axis=1), self.W0)+self.b0),dim=1)

        
        drug_representation = drug_vector1 
        protein_representation = protein_vector1 
        disease_representation = disease_vector1 
        sideeffect_representation = sideeffect_vector1 

        #reconstructing networks
        drug_drug_reconstruct = bi_cul(drug_representation,drug_representation, *self.ddr)
        drug_drug_reconstruct_loss = torch.sum(torch.multiply((drug_drug_reconstruct-drug_drug), (drug_drug_reconstruct-drug_drug)))

        drug_chemical_reconstruct = bi_cul(drug_representation,drug_representation, *self.dcr)
        drug_chemical_reconstruct_loss = torch.sum(torch.multiply((drug_chemical_reconstruct-drug_chemical), (drug_chemical_reconstruct-drug_chemical)))

        drug_disease_reconstruct = bi_cul(drug_representation,disease_representation, *self.ddir)
        drug_disease_reconstruct_loss = torch.sum(torch.multiply((drug_disease_reconstruct-drug_disease), (drug_disease_reconstruct-drug_disease)))

        drug_sideeffect_reconstruct = bi_cul(drug_representation,sideeffect_representation, *self.dsr)
        drug_sideeffect_reconstruct_loss = torch.sum(torch.multiply((drug_sideeffect_reconstruct-drug_sideeffect), (drug_sideeffect_reconstruct-drug_sideeffect)))

        protein_protein_reconstruct = bi_cul(protein_representation,protein_representation, *self.ppr)
        protein_protein_reconstruct_loss = torch.sum(torch.multiply((protein_protein_reconstruct-protein_protein), (protein_protein_reconstruct-protein_protein)))


        protein_sequence_reconstruct = bi_cul(protein_representation,protein_representation, *self.psr)
        protein_sequence_reconstruct_loss = torch.sum(torch.multiply((protein_sequence_reconstruct-protein_sequence), (protein_sequence_reconstruct-protein_sequence)))

        protein_disease_reconstruct = bi_cul(protein_representation,disease_representation, *self.pdir)
        protein_disease_reconstruct_loss = torch.sum(torch.multiply((protein_disease_reconstruct-protein_disease), (protein_disease_reconstruct-protein_disease)))


        drug_protein_reconstruct = bi_cul(drug_representation,protein_representation, *self.dpr)
        tmp = torch.multiply(drug_protein_mask, (drug_protein_reconstruct-drug_protein))
        drug_protein_reconstruct_loss = torch.sum(torch.multiply(tmp, tmp)) #/ (torch.sum(self.drug_protein_mask)

        
        for param in model.parameters():
            l2_loss += torch.norm(param, 2)
        
        loss = drug_protein_reconstruct_loss + 1.0*(drug_drug_reconstruct_loss+drug_chemical_reconstruct_loss+
                                                            drug_disease_reconstruct_loss+drug_sideeffect_reconstruct_loss+
                                                            protein_protein_reconstruct_loss+protein_sequence_reconstruct_loss+
                                                            protein_disease_reconstruct_loss) + l2_loss
        return loss, drug_protein_reconstruct_loss, drug_protein_reconstruct

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

lr = 0.001
model = cuda(Model())
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_and_evaluate(DTItrain, DTIvalid, DTItest, round, verbose=True, num_steps = 4000):
    drug_protein = np.zeros((num_drug,num_protein))
    mask = np.zeros((num_drug,num_protein))
    for ele in DTItrain:
        drug_protein[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein,False)
    protein_drug_normalize = row_normalize(protein_drug,False)

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
                                    drug_drug_, drug_drug_normalize_,\
                                    drug_chemical_, drug_chemical_normalize_,\
                                    drug_disease_, drug_disease_normalize_,\
                                    drug_sideeffect_, drug_sideeffect_normalize_,\
                                    protein_protein_, protein_protein_normalize_,\
                                    protein_sequence_, protein_sequence_normalize_,\
                                    protein_disease_, protein_disease_normalize_,\
                                    drug_encoder_data_, protein_encoder_data_,\
                                    disease_drug_, disease_drug_normalize_,\
                                    disease_protein_, disease_protein_normalize_,\
                                    sideeffect_drug_, sideeffect_drug_normalize_,\
                                    drug_protein_, drug_protein_normalize_,\
                                    protein_drug_, protein_drug_normalize_,\
                                    mask_)
        tloss.backward(retain_graph=True)
        #every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
        if i % 25 == 0 and verbose == True:
        #if i == 0:
            print ('step',i,'total and dtiloss',tloss, dtiloss)

            pred_list = []
            ground_truth = []
            ii = 1
            for ele in DTIvalid:
                print('Evaluating %s / %s' % (ii, len(DTIvalid)))
                pred_list.append(results[ele[0],ele[1]].cpu().detach().numpy())
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
                    pred_list.append(results[ele[0],ele[1]].cpu().detach().numpy())
                    ground_truth.append(ele[2])
                    ii += 1
                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
            print ('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)

    return best_valid_auc, best_valid_aupr, test_auc, test_aupr


test_auc_round = []
test_aupr_round = []
for r in range(10):
    print ('sample round',r+1)
    if opts.t == 'o':
        dti_o = np.loadtxt(network_path+'mat_drug_protein.txt')
    else:
        dti_o = np.loadtxt(network_path+'mat_drug_protein_'+opts.t+'.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i,j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i,j])


    if opts.r == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
    elif opts.r == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_negative_index),replace=False)
    else:
        print ('wrong positive negative ratio')
        break

    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1



    if opts.t == 'unique':
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i,j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i,j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=10*len(whole_positive_index_test),replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=whole_negative_index_test,replace=False)
        else:
            print ('wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test)+len(whole_positive_index_test),3),dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1

        DTItrain = data_set
        DTItest = data_set_test
        rs = np.random.randint(0,1000,1)[0]
        DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
        v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, round = 'unique'+str(r), num_steps=3000)

        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)
        np.savetxt('test_auc_unique_%s' % (r), test_auc_round)
        np.savetxt('test_aupr_unique_%s'% (r), test_aupr_round)

    else:
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0,1000,1)[0]
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs).split(data_set[:,:2], data_set[:,2])
        cv = 0
        for train_index, test_index in kf:
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
            print('doing cv %s round %s' % (cv, r))
            v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, 
                round = str(r) + 'cv_' + str(cv), num_steps=3000)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)
            cv += 1

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        np.savetxt('test_auc_%s' % (r), test_auc_round)
        np.savetxt('test_aupr_%s'% (r), test_aupr_round)
