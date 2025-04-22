import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from attack.metric.metric import all_metrics_for_three_class, save_metric
from attack.utils import baseline_prep, attack_feature_base, adjust_ratio_samples
from parameter_parser import parameter_parser
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
import os
from model.DNN import DNN
import pandas as pd
import scipy

class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list, device):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2
        torch.manual_seed(3407)

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
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



def TC_MIA(args):


    # 1. Training

    print(f"{args['U_method']}-{args['net_name']}-{args['dataset_name']}-----")
    # prep data
    Pin, Pout, Pin_test, Pout_test, mem_train, mem_test,train_sample_label,test_sample_label = baseline_prep(args)


    # target_ratio=(unseen, forget, retain), for the parameter expermients
    #(2:1:2),(2:1:4),(4:1:2)
   # mem_train,Pin,Pout,train_sample_label=adjust_ratio_samples(mem_train,Pin,Pout,train_sample_label,target_ratio=(2, 1, 2))


    # drop the second dimension
    Pin = Pin[:, 0, :]
    Pout = Pout[:, 0, :]
    Pin_test = Pin_test[:, 0, :]
    Pout_test = Pout_test[:, 0, :]

    ACC_list_list=[]
    PRE_list_List=[]
    REC_list_list=[]
    F1_list_list=[]
    AUC_list_list=[]
    TPR001_list_list=[]
    TPR0005_list_list=[]
    TPR005_list_list=[]
    Micro_F1_list=[]


    for method in ["CDS", 'CT']:


        attack_X = attack_feature_base(Pin, Pout,train_sample_label,method)
        ytest = mem_train

        Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

        clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=args['base_num_class'], layer_list=[32, 16], device=torch.device(args['device']))

        clf.criterion = torch.nn.CrossEntropyLoss()
        clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

        clf.to(torch.device(args['device']))

        clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=500, max_patient=200)

        # 2. Testing
        attack_X_test = attack_feature_base(Pin_test, Pout_test,test_sample_label,method)
        y_target = mem_test

        probs = clf.pred(attack_X_test)
        y_pred = np.argmax(probs, axis=1)


        Micro_F1,ACC_list,PRE_list,REC_list,F1_list,AUC_list,TPR001_list,TPR0005_list,TPR005_list = all_metrics_for_three_class(y_target, y_pred, probs)

        ACC_list_list.append(ACC_list)
        PRE_list_List.append(PRE_list)
        REC_list_list.append(REC_list)
        F1_list_list.append(F1_list)
        Micro_F1_list.append(Micro_F1)
        AUC_list_list.append(AUC_list)
        TPR001_list_list.append(TPR001_list)
        TPR0005_list_list.append(TPR0005_list)
        TPR005_list_list.append(TPR005_list)

    save_metric(Micro_F1_list,ACC_list_list,PRE_list_List,REC_list_list,F1_list_list,AUC_list_list,TPR001_list_list,TPR0005_list_list,TPR005_list_list,args)



