import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL.ImImagePlugin import number
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

from attack.metric.metric import all_metrics_for_three_class, save_metric
from attack.utils import baseline_prep, baseline_prep_for_double_attack
from parameter_parser import parameter_parser
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
import os
from model.DNN import DNN
import pandas as pd
from scipy.special import softmax


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list, device):
        super(MLP2Layer, self).__init__()
        torch.manual_seed(3407)

        self.fc1 = nn.Linear(in_dim, layer_list)
        self.fc2 = nn.Linear(layer_list, out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()

        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def train_epochs(self, train_x, train_y, val_x, val_y, epochs, max_patient):
        pbar = tqdm(range(epochs), leave=True, desc=f"Attack Training")
        if val_x is None:
           val_x = train_x
           val_y = train_y
        opt_loss = 1e10
        patient = max_patient
        for i in pbar:
            self.train_one_epoch(train_x, train_y)
            train_loss, train_acc = self.loss_acc(train_x, train_y)
            val_loss, val_acc = self.loss_acc(val_x, val_y)

            pbar.set_postfix({'Loss': train_loss,
                              'Acc': train_acc,
                              'Val Loss': val_loss,
                              'Val Acc': val_acc})
            if opt_loss / 1.001 > val_loss:
                opt_loss = val_loss
                patient = max_patient
            else:
                patient = patient - 1

            if patient == 0:
                pbar.close()
                break

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)
        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def acc(self, X_target, y_target):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        acc = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        return acc

    def pred(self, X_target):
        '''Predict posteriors'''
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        return outputs_target.detach().numpy()




def attack_feature_base(P_shadow, label_list):
    attack_feature = []
    # for posterior_shadow, label in zip(P_shadow, label_list):
    #     attack_feature.append([posterior_shadow[label]])
    for posterior_shadow, label in zip(P_shadow, label_list):
        top_3 = sorted(posterior_shadow, reverse=True)[:3]
        attack_feature.append(top_3)


    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)
    return attack_feature

def Double_Attack(args):


    ACC_list_list = []
    PRE_list_List = []
    REC_list_list = []
    F1_list_list = []
    AUC_list_list = []
    TPR001_list_list = []
    TPR0005_list_list = []
    TPR005_list_list = []
    Micro_F1_list=[]
    # 1. Training
    # prep data
    P_shadow_original,P_shadow_unlearned, P_original_target, P_unlearned_target, mem_train, mem_test,train_sample_label,test_sample_label = baseline_prep_for_double_attack(args)
    # drop the second dimension
    P_shadow_original = P_shadow_original[:, 0, :]
    P_shadow_unlearned = P_shadow_unlearned[:, 0, :]

    P_original_target = P_original_target[:, 0, :]
    P_unlearned_target = P_unlearned_target[:, 0, :]

    #first attack model
    attack_X = attack_feature_base(P_shadow_original,train_sample_label)
    ytest = mem_train
    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

    clf.to(torch.device(args['device']))

    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)

    # Testing original model
    attack_X_original_test = attack_feature_base(P_original_target,test_sample_label)

    probs_original = clf.pred(attack_X_original_test)
    y_pred_original = np.argmax(probs_original, axis=1)

    #second attack model
    attack_X = attack_feature_base(P_shadow_unlearned, train_sample_label)
    ytest = mem_train
    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))

    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

    clf.to(torch.device(args['device']))

    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)

    # Testing unlearned model
    attack_X_unlearned_test2 = attack_feature_base(P_unlearned_target,test_sample_label)
    probs_unlearned = clf.pred(attack_X_unlearned_test2)
    y_pred_unlearned = np.argmax(probs_unlearned, axis=1)

    y_pred = (y_pred_original == 0) * ((y_pred_unlearned == 0) * 0 + (y_pred_unlearned == 1) * 3) + (y_pred_original == 1) * ((y_pred_unlearned == 0) * 1 + (y_pred_unlearned == 1) * 2)
    y_target=mem_test


    probs = np.zeros((probs_original.shape[0],  4))

    probs[:,  0] = probs_original[:, 0] * probs_unlearned[:, 0]
    probs[:,  1] = probs_original[:, 1] * probs_unlearned[:, 0]
    probs[:,  2] = probs_original[:, 0] * probs_unlearned[:, 1]
    probs[:,  3] = probs_original[:, 1] * probs_unlearned[:, 1]

    Micro_F1, ACC_list, PRE_list, REC_list, F1_list, AUC_list, TPR001_list, TPR0005_list, TPR005_list = all_metrics_for_three_class(
        y_target, y_pred, probs)

    ACC_list_list.append(ACC_list)
    PRE_list_List.append(PRE_list)
    REC_list_list.append(REC_list)
    F1_list_list.append(F1_list)
    Micro_F1_list.append(Micro_F1)
    AUC_list_list.append(AUC_list)
    TPR001_list_list.append(TPR001_list)
    TPR0005_list_list.append(TPR0005_list)
    TPR005_list_list.append(TPR005_list)

    save_metric(Micro_F1_list, ACC_list_list, PRE_list_List, REC_list_list, F1_list_list, AUC_list_list,
                TPR001_list_list, TPR0005_list_list, TPR005_list_list, args)


