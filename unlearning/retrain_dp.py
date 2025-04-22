import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
from model.DNN import DNN
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from model.ResNet import resnet18, resnet18_dp
from parameter_parser import parameter_parser
from unlearning.utils import sample_target_samples, save_output
from utils.compute_dp_sgd import apply_dp_sgd_analysis
from utils.dp_optimizer import get_dp_optimizer
from utils.sampling import get_data_loaders_possion
import torch.optim as optim
from torch.utils.data import TensorDataset
from opacus.validators import ModuleValidator
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 获取当前时间并格式化为唯一字符串
def retrain_dp(args):
    retrain_dp_save_target_for_population_attack(args)


def retrain_dp_save_target_for_population_attack(args):

    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=args['batch_size'],microbatch_size=1,iterations=1)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=False)
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    C=args['C']
    delta=5*1e-4
    sigma=args['sigma']

    original_model = DNN(args)
    original_model=original_model.to(args['device'])
    optimizer = get_dp_optimizer(lr=args['lr'], C_t=C, sigma=sigma, batch_size=args['batch_size'], model=original_model)
    optimizer.minibatch_size = args['batch_size']
    flag=f'dp_{sigma}'
    for epoch in range(args['num_epochs']):
        epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(target_m), sigma, (epoch+1)*len(train_loader), orders, delta) #comupte privacy cost


        train_with_dp(original_model, train_loader, optimizer, args['device'])
        test_loss, test_accuracy =validation(original_model, test_loader,args['device'])
        train_loss, train_acc =validation(original_model, train_loader,args['device'])

        print(
            f'epoches:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

    #unlearned model
    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        target_sample, remaining_data = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)

        unlearned_model = DNN(args)
        unlearned_model = unlearned_model.to(args['device'])
        optimizer = get_dp_optimizer(lr=args['lr'], C_t=C, sigma=sigma, batch_size=args['batch_size'], model=unlearned_model)
        remaining_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        for epoch in range(args['num_epochs']):
            epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(target_m), sigma, (epoch+1)*len(train_loader), orders,
                                                        delta)  # comupte privacy cost

            train_with_dp(unlearned_model, remaining_loader, optimizer, args['device'])
            test_loss, test_accuracy = validation(unlearned_model, test_loader, args['device'])
            train_loss, train_acc = validation(unlearned_model, remaining_loader, args['device'])
            print(
                f'epoch:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

        save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{args['sigma']}/target/{t}/"
        os.makedirs(save_path, exist_ok=True)

        save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


def train(epoch,model, train_loader, test_loader,optimizer,privacy_engine,args):
    device= args['device']
    model=model.to(device)
    test_acc=0.
    train_acc=0.
    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args['batch_size'],
            optimizer=optimizer
    ) as new_train_loader:
        for id, (data, target) in enumerate(new_train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output =model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            loss.backward()
            optimizer.step()
        train_acc = test_model_acc(model,train_loader,device)
        test_acc = test_model_acc(model,test_loader,device)
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        print(f' epoch:{epoch} |(ε = {epsilon:.2f}, δ = 1e-5) | train acc:{round(train_acc, 4)} | test acc: {round(test_acc, 4)}')


def test_model_acc(model, test_loader,device):
    model.eval()
    model = model.to(device)
    correct = 0

    with torch.no_grad():

        for data, target  in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data).to(device)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / len(test_loader.dataset)

def train_with_dp(model, train_loader, optimizer, device):
    model=model.to(device)
    correct = 0
    for id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape) == 2:
                output = torch.squeeze(output, 0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()

def validation(model, test_loader,device):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum')

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc