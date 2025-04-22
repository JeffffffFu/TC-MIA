from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model.DNN import DNN
from unlearning.utils import sample_target_samples, l1_regularization, pruning_model, check_sparsity, extract_mask, \
    remove_prune, prune_model_custom, save_output
import os
from copy import deepcopy
import copy

#paper <Model sparsification can simplify machine unlearning>
def sparsity(args):
     sparsity_save_target_for_population_attack(args)
     sparsity_save_shadow_for_population_attack(args)


def sparsity_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
   # original_model.load_state_dict(torch.load(f"{args['dataset_name']}_{args['net_name']}_original_model.pth"))
   # original_model.to(args['device'])
    initial_model=copy.deepcopy(original_model)
    # prune model
    prune=0.8
    original_model = pruning_model(original_model, prune)
    remain_weight = check_sparsity(original_model)

    # after pruning
    current_mask = extract_mask(original_model.state_dict())
    remove_prune(original_model)
    pruned_model = prune_model_custom(original_model, current_mask, args)


    for t in range(args['trials']):
        print(f'The {t}-th trails')
        # prune model
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(pruned_model.state_dict(), strict=False)

        #unlearned model
        forget_set,retain_set=sample_target_samples(target_m,args['proportion_of_group_unlearn'],args['dataset_name'])
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=True)
        unlearned_model=sparsity_train(unlearned_model, retain_loader,forget_loader, test_loader, args)

        save_output('target', args, initial_model, unlearned_model, forget_set, retain_set, test_data,shadow_um, t)




def sparsity_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    initial_model=copy.deepcopy(original_model)

    prune=0.9
    original_model = pruning_model(original_model, prune)
    remain_weight = check_sparsity(original_model)

    # after pruning
    current_mask = extract_mask(original_model.state_dict())
    remove_prune(original_model)
    pruned_model = prune_model_custom(original_model, current_mask, args)

    for t in range(args['observations']):
        print(f'The {t}-th observations')
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(pruned_model.state_dict(), strict=False)

        # unlearned model
        forget_set, retain_set = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'], args['dataset_name'])
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=True)
        unlearned_model=sparsity_train(unlearned_model, retain_loader,forget_loader, test_loader, args)
        save_output('shadow', args, initial_model, unlearned_model, forget_set, retain_set, test_data,shadow_um, t)




def sparsity_train(unlearned_model,retain_loader,forget_loader,test_loader,args,with_l1=False):

    optimizer = optim.Adam(unlearned_model.parameters(), 0.0005, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    no_l1_epochs=0
    sparse_scheduler= "increase"
    sno_l1_epochs=0
    alpha=0.0005
    unlearned_model.train()

    for epoch in range(num_epochs):
        losses = []
        current_alpha=0.
        if epoch <= num_epochs - no_l1_epochs:
            if sparse_scheduler == 'decay':
                current_alpha = (2 - (2 * epoch / num_epochs)) * alpha
            elif sparse_scheduler == 'constant':
                current_alpha = alpha
            elif sparse_scheduler == 'increase':
                current_alpha = (2 * epoch / num_epochs) * alpha
        elif epoch > num_epochs - sno_l1_epochs:
            current_alpha = 0

        for data, target  in retain_loader:

            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = unlearned_model.forward_propagation(data)
            loss = criterion(output, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(unlearned_model)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return unlearned_model