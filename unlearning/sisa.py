import pandas as pd
from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model.DNN import DNN
from unlearning.utils import sample_target_samples
import os
from torch.utils.data import DataLoader, Subset
import random
from torch.utils.data import ConcatDataset

def sisa(args):
     sisa_save_target_for_population_attack(args)
     sisa_save_shadow_for_population_attack(args)


def sisa_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])
    print("dataset and net_name:",args['dataset_name'],args['net_name'])

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])
    num_shards=3
    shards,shard_indices_list=shard_dataset(target_m, num_shards)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model_dict = {}

    for id,shard in enumerate(shards):
        train_loader_shards = torch.utils.data.DataLoader(
            shard, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader_shards, test_loader)
        original_model_dict[f"{id}"] = original_model



    for t in range(args['trials']):
        print(f'The {t}-th trails')
        shard_id=random.randint(0, num_shards-1)
        #unlearned model
        forget_set, retain_set = sample_target_samples(shards[shard_id], args['proportion_of_group_unlearn'], args['dataset_name'],False)

        #retrain
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        shard_id_unlearned_model = DNN(args)
        shard_id_unlearned_model.train_model( retain_loader, test_loader, args)

        save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/"
        os.makedirs(save_path, exist_ok=True)

        print("unlearned sample:------------------")
        for i, idx in enumerate(forget_set.indices):
            data, label = forget_set.dataset[idx]
            save_path_target_sample = f'{save_path}/{i}'
            os.makedirs(save_path_target_sample, exist_ok=True)
            unlearned_sample_original_model_posterior, unlearned_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)

            torch.save(unlearned_sample_original_model_posterior,
                       f"{save_path_target_sample}/unlearned_sample_original_model_posterior.pth")
            torch.save(unlearned_sample_unlearned_model_posterior,
                       f"{save_path_target_sample}/unlearned_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_target_sample}/target_sample_label.pth")

        for id, shard in enumerate(shards):
            if id!=shard_id:
                retain_set = ConcatDataset([retain_set, shard])

        retain_sample, _ = sample_target_samples(retain_set,  int(len(forget_set)), args['dataset_name'])

        for i, idx in enumerate(retain_sample.indices):
            data, label = retain_sample.dataset[idx]
            save_path_retain_sample = f'{save_path}/{i}'
            if not os.path.exists(save_path_retain_sample):
                os.makedirs(save_path_retain_sample)
            retain_sample_original_model_posterior, retain_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)
            torch.save(retain_sample_original_model_posterior,
                       f"{save_path_retain_sample}/retain_sample_original_model_posterior.pth")
            torch.save(retain_sample_unlearned_model_posterior,
                       f"{save_path_retain_sample}/retain_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_retain_sample}/retain_sample_label.pth")

        unseen_sample, _ = sample_target_samples(test_data, int(len(forget_set)), args['dataset_name'])

        for i, idx in enumerate(unseen_sample.indices):
            data, label = unseen_sample.dataset[idx]
            save_path_unseen_sample = f'{save_path}/{i}'
            if not os.path.exists(save_path_unseen_sample):
                os.makedirs(save_path_unseen_sample)
            unseen_sample_original_model_posterior, unseen_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)
            torch.save(unseen_sample_original_model_posterior,
                       f"{save_path_unseen_sample}/unseen_sample_original_model_posterior.pth")
            torch.save(unseen_sample_unlearned_model_posterior,
                       f"{save_path_unseen_sample}/unseen_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_unseen_sample}/unseen_sample_label.pth")


        sisa_save_model_performance(original_model_dict, shard_id, shard_id_unlearned_model, forget_set, retain_set,
                               test_data,num_shards,args)

def sisa_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    num_shards = 3
    shards, shard_indices_list = shard_dataset(shadow_m, num_shards)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model_dict = {}

    for id, shard in enumerate(shards):
        train_loader_shards = torch.utils.data.DataLoader(
            shard, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader_shards, test_loader)
        original_model_dict[f"{id}"] = original_model
    print("dataset and net_name:",args['dataset_name'],args['net_name'])

    for t in range(args['observations']):
        print(f'The {t}-th observations')
        shard_id = random.randint(0, num_shards - 1)

        # unlearned model
        forget_set, retain_set = sample_target_samples(shards[shard_id], args['proportion_of_group_unlearn'], args['dataset_name'])

        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        shard_id_unlearned_model = DNN(args)
        shard_id_unlearned_model.train_model(retain_loader, test_loader, args)

        save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/shadow/{t}/"
        os.makedirs(save_path, exist_ok=True)

        print("unlearned sample:------------------")
        for i, idx in enumerate(forget_set.indices):
            data, label = forget_set.dataset[idx]
            save_path_target_sample = f'{save_path}/{i}'
            os.makedirs(save_path_target_sample, exist_ok=True)
            unlearned_sample_original_model_posterior, unlearned_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)

            torch.save(unlearned_sample_original_model_posterior,
                       f"{save_path_target_sample}/unlearned_sample_original_model_posterior.pth")
            torch.save(unlearned_sample_unlearned_model_posterior,
                       f"{save_path_target_sample}/unlearned_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_target_sample}/target_sample_label.pth")


        for id, shard in enumerate(shards):
            if id!=shard_id:
                retain_set = ConcatDataset([retain_set, shard])

        retain_sample, _ = sample_target_samples(retain_set,  int(len(forget_set)),
                                                 args['dataset_name'])

        for i, idx in enumerate(retain_sample.indices):
            data, label = retain_sample.dataset[idx]
            save_path_retain_sample = f'{save_path}/{i}'
            if not os.path.exists(save_path_retain_sample):
                os.makedirs(save_path_retain_sample)
            retain_sample_original_model_posterior, retain_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)
            torch.save(retain_sample_original_model_posterior,
                       f"{save_path_retain_sample}/retain_sample_original_model_posterior.pth")
            torch.save(retain_sample_unlearned_model_posterior,
                       f"{save_path_retain_sample}/retain_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_retain_sample}/retain_sample_label.pth")

        unseen_sample, _ = sample_target_samples(shadow_um,  int(len(forget_set)), args['dataset_name'])

        for i, idx in enumerate(unseen_sample.indices):
            data, label = unseen_sample.dataset[idx]
            save_path_unseen_sample = f'{save_path}/{i}'
            if not os.path.exists(save_path_unseen_sample):
                os.makedirs(save_path_unseen_sample)
            unseen_sample_original_model_posterior, unseen_sample_unlearned_model_posterior = \
                sisa_posterior_aggregation(original_model_dict, shard_id, shard_id_unlearned_model, data,
                                           num_shards)
            torch.save(unseen_sample_original_model_posterior,
                       f"{save_path_unseen_sample}/unseen_sample_original_model_posterior.pth")
            torch.save(unseen_sample_unlearned_model_posterior,
                       f"{save_path_unseen_sample}/unseen_sample_unlearned_model_posterior.pth")
            torch.save(label, f"{save_path_unseen_sample}/unseen_sample_label.pth")

        # sisa_save_model_performance(original_model_dict, shard_id, shard_id_unlearned_model, forget_set, retain_set,
        #                        test_data,num_shards,args)

def sisa_posterior_aggregation(original_model_dict,shard_id,shard_id_unlearned_model,target_sample,num_shards):
    original_model_posterior_list = []
    unlearned_model_posterior_list = []

    for id in range(num_shards):
        shard_original_model = original_model_dict[f"{id}"]
        posterior = shard_original_model.predict_proba(target_sample)
        original_model_posterior_list.append(posterior)
        if id != shard_id:
            unlearned_model_posterior_list.append(posterior)
        else:
            posterior = shard_id_unlearned_model.predict_proba(target_sample)
            unlearned_model_posterior_list.append(posterior)

    original_model_posterior = np.mean(original_model_posterior_list, axis=0)
    unlearned_model_posterior = np.mean(unlearned_model_posterior_list, axis=0)
    return original_model_posterior, unlearned_model_posterior


def compute_accuracy_sisa(model_dict, shard_id, unlearned_model, dataset, num_shards):
    correct_original, correct_unlearned = 0, 0
    dataset_size = len(dataset)

    for idx in range(len(dataset)):
        data, label = dataset[idx]
        original_posterior_list, unlearned_posterior_list = [], []

        for id in range(num_shards):
            posterior = model_dict[f"{id}"].predict_proba(data)
            original_posterior_list.append(posterior)
            unlearned_posterior_list.append(posterior if id != shard_id else unlearned_model.predict_proba(data))

        original_pred = np.mean(original_posterior_list, axis=0).argmax()
        unlearned_pred = np.mean(unlearned_posterior_list, axis=0).argmax()

        correct_original += (original_pred == label)
        correct_unlearned += (unlearned_pred == label)

    return correct_original / dataset_size, correct_unlearned / dataset_size


def sisa_save_model_performance(original_model_dict, shard_id, shard_id_unlearned_model, forget_set, retain_set, test_set,
                           num_shards, args):
    save_path = f"{os.getcwd()}/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/"
    os.makedirs(save_path, exist_ok=True)

    forget_acc_orig, forget_acc_unl = compute_accuracy_sisa(original_model_dict, shard_id, shard_id_unlearned_model,
                                                       forget_set, num_shards)
    retain_acc_orig, retain_acc_unl = compute_accuracy_sisa(original_model_dict, shard_id, shard_id_unlearned_model,
                                                       retain_set, num_shards)
    test_acc_orig, test_acc_unl = compute_accuracy_sisa(original_model_dict, shard_id, shard_id_unlearned_model, test_set,
                                                   num_shards)

    metrics_dict = {
        "forget_set_acc_original": round(forget_acc_orig, 4),
        "retain_set_acc_original": round(retain_acc_orig, 4),
        "test_acc_original": round(test_acc_orig, 4),
        "forget_set_acc_unlearned": round(forget_acc_unl, 4),
        "retain_set_acc_unlearned": round(retain_acc_unl, 4),
        "test_acc_unlearned": round(test_acc_unl, 4),
    }

    pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"]).to_csv(f"{save_path}/model_performance.csv",
                                                                                 index=False)

def shard_dataset(dataset, num_shards):
    indices = list(range(len(dataset)))
   # np.random.shuffle(indices)
    shards = []
    shard_size = len(dataset) // num_shards
    shard_indices_list=[]
    for i in range(num_shards):
        shard_indices = indices[i * shard_size: (i + 1) * shard_size]
        shards.append(Subset(dataset, shard_indices))
        shard_indices_list.append(shard_indices)
    return shards,shard_indices_list