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
from tqdm import tqdm
from torch.autograd import grad


def IF(args):
    IF_save_target_for_population_attack(args)
    IF_save_shadow_for_population_attack(args)


def IF_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)


    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        target_sample,remaining_data=sample_target_samples(target_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        target_loader = torch.utils.data.DataLoader(
            target_sample, batch_size=args['batch_size'], shuffle=False)
        remaining_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        retain_loader = torch.utils.data.DataLoader(
            remaining_data.dataset, batch_size=1, shuffle=False
        )
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=IF_train(unlearned_model, target_loader,remaining_loader, retain_loader,test_loader, args)
        # retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        # forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        # test_acc = unlearned_model.test_model_acc(test_loader)
        # print(f'forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))

        save_output('target', args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


def IF_save_shadow_for_population_attack(args):
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

        #unlearned model
        target_sample,remaining_data=sample_target_samples(shadow_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        target_loader = torch.utils.data.DataLoader(
            target_sample, batch_size=args['batch_size'], shuffle=False)
        remaining_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        retain_loader = torch.utils.data.DataLoader(
            remaining_data.dataset, batch_size=1, shuffle=False
        )
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=IF_train(unlearned_model, target_loader,remaining_loader, retain_loader,test_loader, args)
        unlearned_model.test_model_acc(test_loader)

        save_output('shadow', args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


def IF_train(unlearned_model,forget_loader,retain_grad_loader,retain_loader,test_loader,args):
    params = []
    for param in unlearned_model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(args['device'])
    retain_grad = torch.zeros_like(torch.cat(params)).to(args['device'])

    total = 0
    unlearned_model.eval()
    criterion = nn.CrossEntropyLoss()

    for i, (data, label) in enumerate(tqdm(forget_loader)):
        unlearned_model.zero_grad()
        real_num = data.shape[0]
        data = data.to(args['device'])
        label = label.to(args['device'])
        output = unlearned_model.forward_propagation(data)
        loss = criterion(output, label)
        f_grad = sam_grad(unlearned_model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, (data, label) in enumerate(tqdm(retain_grad_loader)):
        unlearned_model.zero_grad()
        real_num = data.shape[0]
        data = data.to(args['device'])
        label = label.to(args['device'])
        output = unlearned_model.forward_propagation(data)
        loss = criterion(output, label)
        r_grad = sam_grad(unlearned_model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2
    perturb = woodfisher(
        unlearned_model,
        retain_loader,
        device=args['device'],
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    apply_perturb(unlearned_model, 0.2 * perturb)  #alpha=0.2

    return unlearned_model

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data
        curr += length

def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = len(train_dl)

    for idx, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        output = model.forward_propagation(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        if idx == 0:
            o_vec = torch.clone(sample_grad)
        else:
            tmp = torch.dot(o_vec, sample_grad)
            k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
            o_vec -= (tmp / (N + tmp)) * o_vec
    return k_vec
