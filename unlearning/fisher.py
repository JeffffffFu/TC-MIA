from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output
import os

def fisher(args):
    fisher_save_target_for_population_attack(args)
    fisher_save_shadow_for_population_attack(args)


def fisher_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m,  shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
  #  original_model.load_state_dict(torch.load(f"{args['dataset_name']}_original_model.pth", map_location=args['device']))
   # original_model.to(args['device'])

    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        target_sample,remaining_data=sample_target_samples(target_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        retain_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        forget_loader = torch.utils.data.DataLoader(
            target_sample, batch_size=args['batch_size'], shuffle=False)
        unlearned_model=fisher_train(unlearned_model, retain_loader, test_loader, args)

        retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (t,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))

        save_output('target', args, original_model, unlearned_model, target_sample, remaining_data, test_data,shadow_um,t)


def fisher_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    for t in range(args['observations']):
        print(f'The {t}-th observations')
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        # unlearned model
        target_sample,remaining_data=sample_target_samples(shadow_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        retain_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        unlearned_model=fisher_train(unlearned_model, retain_loader, test_loader, args)


        unlearned_model.test_model_acc(test_loader)

        save_output('shadow', args, original_model, unlearned_model, target_sample, remaining_data, test_data,shadow_um,t)



import copy
import tqdm
def fisher_train(unlearned_model,train_loader,test_loader,args):
    criterion = nn.CrossEntropyLoss()
    for p in unlearned_model.parameters():
        p.data0 = copy.deepcopy(p.data.clone())
    hessian(train_loader, unlearned_model, criterion, args)
    for i, p in enumerate(unlearned_model.parameters()):
        mu, var = get_mean_var(p, args, False)
        mu = mu.to(args['device'])
        var = var.to(args['device'])
        p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
    return unlearned_model

def hessian(train_loader, model, loss_fn, args):
    model.eval()
    device = args['device']
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
  #  train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0

    for data, orig_target in train_loader:
        data, orig_target = data.to(device), orig_target.to(device)
        output = model.forward_propagation(data)
        prob = torch.nn.functional.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        p.grad2_acc /= len(train_loader)


def get_mean_var(p, args, is_base_dist=False):
    alpha=1e-6
    if args['dataset_name'] == 'cifar100':
        num_classes = 100
    elif args['dataset_name'] == 'celebA':
        num_classes = 40
    else:
        num_classes = 10
    var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.shape[0] == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())

    # if p.shape[0] == args.num_classes and (
    #     (args.num_indexes_to_replace == 4500 and args.dataset == "cifar10")
    #     or (args.num_indexes_to_replace == 450 and args.dataset == "cifar100")
    # ):
    #     mu[args.class_to_replace] = 0
    #     var[args.class_to_replace] = 0.0001
    if p.shape[0] == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
    return mu, var
