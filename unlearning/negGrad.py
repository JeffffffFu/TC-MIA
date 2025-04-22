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
from itertools import cycle

def NegGrad(args):
     NegGrad_save_target_for_population_attack(args)
     NegGrad_save_shadow_for_population_attack(args)


def NegGrad_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    # original_model.load_state_dict(torch.load(f"{args['dataset_name']}_original_model.pth", map_location=args['device']))
    # original_model.to(args['device'])
    original_model.train_model(train_loader, test_loader)


    for t in range(args['trials']):
        print(f'The {t}-th trails')
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)

        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False)
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (
            t, round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))
        unlearned_model=NegGrad_train(original_model, unlearned_model, forget_loader,retain_loader, test_loader, args)
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



def NegGrad_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])
    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    for t in range(args['observations']):
        print(f'The {t}-th observations')

        # unlearned model
        forget_set, retain_set = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'], args['dataset_name'])

        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False)
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=NegGrad_train(original_model, unlearned_model, forget_loader,retain_loader, test_loader, args)
        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss

def NegGrad_train(original_model,unlearned_model,forget_loader,retain_loader,test_loader,args):
    alpha= 0.6  #0.99 and lr=0.001 unlearn_epoch=20 good for  TinyImageNet and CIFAR-100
    # 0.96 and lr=0.001 unlearn_epoch=10 good for cifar10
    unlearn_epoch=40

    weight_decay=5e-4
    optimizer= optim.Adam(unlearned_model.parameters(), lr=0.001, weight_decay=5e-4)
    # if args['dataset_name']=='cifar10' :
    #     alpha = 0.8
    #     unlearn_epoch = 10
    # elif  args['dataset_name']=='cinic10':
    #     alpha = 0.9
    #     unlearn_epoch = 20
  #  optimizer= optim.SGD(unlearned_model.parameters(), lr=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unlearn_epoch)

    criterion = nn.CrossEntropyLoss()

    unlearned_model.train()

    for t in range(unlearn_epoch):
        for retain_batch, forget_batch in zip(retain_loader,cycle(forget_loader)):
            retain_batch = [tensor.to(args['device']) for tensor in retain_batch]
            forget_batch = [tensor.to(args['device']) for tensor in forget_batch]
            inputs_r, targets_r = retain_batch
            inputs_f, targets_f = forget_batch
            optimizer.zero_grad()
            outputs_r = unlearned_model.forward_propagation(inputs_r)
            outputs_f = unlearned_model.forward_propagation(inputs_f)

            loss_r = criterion(outputs_r, targets_r)
            loss_f =criterion(outputs_f, targets_f)
          #  l2_loss=l2_penalty(unlearned_model, original_model, weight_decay)
            # print("loss_r:",loss_r)
            # print("loss_f",loss_f)
            # print("l2_loss",l2_loss)
            loss = (alpha * loss_r) - (1 - alpha) * loss_f
           # print("loss",loss)
            if loss<0:
                alpha=1.0

           # loss = loss_r
            unlearned_model.zero_grad()
            loss.backward()
            optimizer.step()
          #  scheduler.step()

        retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (
        t, round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))
    return unlearned_model