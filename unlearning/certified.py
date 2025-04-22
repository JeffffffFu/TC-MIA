import math

import pandas as pd
from mpmath import sqrtm
from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model.DNN import DNN
from unlearning.utils import sample_target_samples, get_gradient_norm, save_output
import os

from torch.autograd import grad
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

def certified(args):
    certified_save_target_for_population_attack(args)
   # certified_save_shadow_for_population_attack(args)


def certified_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    G=get_gradient_norm(original_model,target_m,args)

    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        target_sample,remaining_data=sample_target_samples(target_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        target_loader = torch.utils.data.DataLoader(
            target_sample, batch_size=args['batch_size'], shuffle=False)
        target_sample_label=target_sample.dataset[target_sample.indices[0]][1]
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())

        unlearned_model,test_acc=certified_train(unlearned_model, target_loader, remaining_data, test_loader,G, args)

        save_output('target', args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


def certified_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m,  shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    G=get_gradient_norm(original_model,shadow_m,args)

    for t in range(args['observations']):
        print(f'The {t}-th observations')

        # unlearned model
        target_sample, remaining_data = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'], args['dataset_name'])
        target_loader = torch.utils.data.DataLoader(
            target_sample, batch_size=args['batch_size'], shuffle=False)
        target_sample_label = target_sample.dataset[target_sample.indices[0]][1]
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=certified_train(unlearned_model, target_loader, remaining_data, test_loader, G,args)

        save_output('shadow', args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)



def certified_train(unlearned_model,train_loader,res_set,test_loader,G,args):
    weight_decay,gamma,s1, s2, scale=0.0005,0.01,10,1000,1000.
    eps=args['eps']
    delta=0.1
    print("G:",G)
    bound=get_upper_bound(G,args)
    print('bound:',bound)
    std=get_std(eps,delta,bound)
    print('std:',std)
    g = grad_batch_approx(train_loader, weight_decay, unlearned_model,  args['device'])
    IF = newton_update(g, args['batch_size'], res_set, weight_decay,gamma, unlearned_model, s1, s2, scale, args['device'])
    for i, param in enumerate(unlearned_model.parameters()):
        # param.data.add_(len(unl_set) / len(res_set) * delta[i] + args.std * torch.randn(param.data.size()).to(args.device))
        param.data.add_(-IF[i] + std * torch.randn(param.data.size()).to(args['device']))
    test_acc = unlearned_model.test_model_acc(test_loader)
    print('test acc %s ' % (round(test_acc, 4)))

    return unlearned_model,test_acc

#the batch_loader is retain set
def grad_batch(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    params = [p for p in model.parameters() if p.requires_grad]
    grad_batch = [torch.zeros_like(p).cpu() for p in params]
    num = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        num += targets.shape[0]
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        grad_mini = list(grad(criterion(outputs, targets), params))
        for i in range(len(grad_batch)):
            grad_batch[i] += grad_mini[i].cpu().detach()

    for i in range(len(grad_batch)):
        grad_batch[i] /= num

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    grad_reg = list(grad(lam * l2_reg, params))
    for i in range(len(grad_batch)):
        grad_batch[i] += grad_reg[i].cpu().detach()
    return [p.to(device) for p in grad_batch]



#the batch_loader is unlearned set
def grad_batch_approx(batch_loader,weight_decay, model, device):
    model.eval()
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss += criterion(outputs, targets)

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss + weight_decay * l2_reg, params))


def newton_update(g, batch_size, res_set, lam, gamma, model, s1, s2, scale, device):
    model.eval()
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    H_res = [torch.zeros_like(p) for p in g]
    for i in tqdm(range(s1)):
        H = [p.clone() for p in g]
        sampler = RandomSampler(res_set, replacement=True, num_samples=batch_size * s2)
        # Create a data loader with the sampler
        res_loader = DataLoader(res_set, batch_size=batch_size, sampler=sampler)
        res_iter = iter(res_loader)
        for j in range(s2):
            data, target = next(res_iter)
            data, target = data.to(device), target.to(device)
            z = model(data)
            loss = criterion(z, target)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            # Add L2 regularization to the loss
            loss += (lam + gamma) * l2_reg
            H_s = hvp(loss, params, H)
            with torch.no_grad():
                for k in range(len(params)):
                    H[k] = H[k] + g[k] - H_s[k] / scale
                if j % int(s2 / 10) == 0:
                    print(f'Epoch: {j}, Sum: {sum([torch.norm(p, 2).item() for p in H])}')
        for k in range(len(params)):
            H_res[k] = H_res[k] + H[k] / scale

    return [p / s1 for p in H_res]


def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def get_std(eps,delta,bound):
    std=bound*(math.sqrt(2*math.log(1.25/delta)))
    return std/eps


in_dim = {
    "stl10": 96 * 96 * 3,
    "cifar10": 32 * 32 * 3,
    "cifar100": 32 * 32 * 3,
    "svhn": 32 * 32 * 3,
    "celebA": 128 * 128 * 3,
    "mnist": 28 * 28 * 1,
    "fmnist": 28 * 28 * 1,
}


def get_upper_bound(G,args):
    M=L=1
    lamd_min=0
    lamd = 200
    C=args['max_norm']
    d=in_dim[args['dataset_name']]
    pho=0.1
    if args['net_name']=='resnet18':
        lamd=2000
    a=(2*C*(M*C+lamd)+G)/(lamd+lamd_min)
    b=2*L*C+G
    c=16*math.sqrt(math.log(d)/pho)/(lamd+lamd_min)+1/16
    return a+b*c