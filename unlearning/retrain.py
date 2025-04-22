import os
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from attack.Double_Attack import attack_feature_base, MLP2Layer
from attack.utils import baseline_prep_for_double_attack
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset, split_dataset2, split_dataset3
from model.DNN import DNN
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from scipy.stats import pearsonr

from parameter_parser import parameter_parser
from unlearning.utils import sample_target_samples, save_output, calculate_confidence_with_subsets, TransformedDataset, \
    sample_target_samples2, get_top_bottom_n_indices, add_gaussian_noise


def retrain(args):
     retrain_save_target_for_population_attack_batch(args)
     retrain_save_shadow_for_population_attack_batch(args)

# just save posterior
def retrain_save_target_for_population_attack_batch(args):
    print("dataset and net_name:",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'])

    target_m,shadow_m,shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    acc_train_loader = original_model.test_model_acc(train_loader)
    acc_test_loader = original_model.test_model_acc(test_loader)
    print(acc_train_loader,acc_test_loader)

    for t in range(args['trials']):
        print(f'The {t}-th trails')

        # unlearned model
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        unlearned_model = DNN(args)
        unlearned_model.train_model(retain_loader, test_loader)
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)


def retrain_save_shadow_for_population_attack_batch(args):
    print("dataset and net_name:",args['dataset_name'],args['net_name'])
    train_data, test_data = get_data(args['dataset_name'])

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    for t in range(args['observations']):
        print(f'The {t}-th observations')

        # unlearned model
        forget_set, retain_set = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        unlearned_model = DNN(args)
        unlearned_model.train_model(retain_loader, test_loader)
        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)


