import os

import pandas as pd
from torch.utils.data import DataLoader, Subset
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def sample_target_samples(train_data, proportion, dataset_name ,unlearn_label=False):
    num_unlearn_samples=0
    if unlearn_label:
        target_label = 5
        unlearn_indices = [i for i, (_, label) in enumerate(train_data) if label == target_label]
        selected_samples = Subset(train_data, unlearn_indices)

        all_indices = list(range(len(train_data)))
        remaining_indices = list(set(all_indices) - set(unlearn_indices))
        unlearn_dataset = Subset(train_data, remaining_indices)

    else:
        all_indices = list(range(len(train_data)))
        if proportion <=0:
            raise ValueError("Proportion must be greater than zero")
        elif proportion >=1:
            num_unlearn_samples= int(proportion)
        else:
            num_unlearn_samples = int(len(train_data) * proportion)

        random_indices = random.sample(all_indices, num_unlearn_samples)
        selected_samples = Subset(train_data, random_indices)

        remaining_indices = list(set(all_indices) - set(random_indices))
        unlearn_dataset = Subset(train_data, remaining_indices)

    return selected_samples, unlearn_dataset

# for over-well, return 2 target sets
def sample_target_samples2(train_data_transformed,train_data, proportion, dataset_name ,unlearn_label=False):
    num_unlearn_samples=0
    if unlearn_label:
        target_label = 5
        unlearn_indices = [i for i, (_, label) in enumerate(train_data_transformed) if label == target_label]
        selected_samples_transformed = Subset(train_data_transformed, unlearn_indices)

        all_indices = list(range(len(train_data_transformed)))
        remaining_indices = list(set(all_indices) - set(unlearn_indices))
        unlearn_dataset_transformed = Subset(train_data_transformed, remaining_indices)

    else:
        all_indices = list(range(len(train_data_transformed)))
        if proportion <=0:
            raise ValueError("Proportion must be greater than zero")
        elif proportion >=1:
            num_unlearn_samples= int(proportion)
        else:
            num_unlearn_samples = int(len(train_data_transformed) * proportion)

        random_indices = random.sample(all_indices, num_unlearn_samples)
        selected_samples_transformed = Subset(train_data_transformed, random_indices)

        remaining_indices = list(set(all_indices) - set(random_indices))
        unlearn_dataset_transformed = Subset(train_data_transformed, remaining_indices)

        selected_samples=Subset(train_data, random_indices)
        unlearn_dataset= Subset(train_data, remaining_indices)

    return selected_samples_transformed, unlearn_dataset_transformed,selected_samples,unlearn_dataset






def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def pruning_model(model, px):
    print("Apply Unstructured L1 Pruning Globally (all conv layers)")
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )
    return model

def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = ", 100 * (1 - zero_sum / sum_list), "%")
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratie = None

    return remain_weight_ratie

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def remove_prune(model):
    print("Remove hooks for multiplying masks (all conv layers)")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m, "weight")

def prune_model_custom(model, mask_dict,args):
    print("Pruning with custom mask (all conv layers)")
    model=model.to(args['device'])

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name + ".weight_mask"
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(
                    m, "weight", mask=mask_dict[name + ".weight_mask"]
                )
            else:
                print("Can not find [{}] in mask_dict".format(mask_name))
    return model

def get_gradient_norm(model,target_m,args):
    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=1, shuffle=False)
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    for data, target in train_loader:
        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    param_norm = nn.utils.parameters_to_vector(model.parameters()).norm()

    return param_norm

def save_output(shadow_or_target,args,original_model,unlearned_model,forget_set,retain_set,test_set,shadow_um,t):


    save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{shadow_or_target}/{t}/"
    os.makedirs(save_path, exist_ok=True)
    target_list=['target','well-over','well-well','over-over','over-well','high_conf','low_conf','low_entropy','high_entropy','low_asr','high_asr','random','dp_0.5','dp_1.0','dp_1.5','dp_2.0','in','out']
    print("unlearned sample:------------------")
    for i, idx in enumerate(forget_set.indices):
        data, label = forget_set.dataset[idx]
        save_path_target_sample = f'{save_path}/{i}'
        os.makedirs(save_path_target_sample, exist_ok=True)
        unlearned_sample_original_model_posterior = original_model.predict_proba(data)
        unlearned_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(unlearned_sample_original_model_posterior,
                   f"{save_path_target_sample}/unlearned_sample_original_model_posterior.pth")
        torch.save(unlearned_sample_unlearned_model_posterior,
                   f"{save_path_target_sample}/unlearned_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_target_sample}/target_sample_label.pth")

    print("retain sample:------------------")
    retain_sample, _ = sample_target_samples(retain_set, int(len(forget_set)), args['dataset_name'])

    for i, idx in enumerate(retain_sample.indices):
        data, label = retain_sample.dataset[idx]
        save_path_retain_sample = f'{save_path}/{i}'
        if not os.path.exists(save_path_retain_sample):
            os.makedirs(save_path_retain_sample)
        retain_sample_original_model_posterior = original_model.predict_proba(data)
        retain_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(retain_sample_original_model_posterior,
                   f"{save_path_retain_sample}/retain_sample_original_model_posterior.pth")
        torch.save(retain_sample_unlearned_model_posterior,
                   f"{save_path_retain_sample}/retain_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_retain_sample}/retain_sample_label.pth")

    print("unseen sample:------------------")
    if shadow_or_target in target_list:
        unseen_sample, _ = sample_target_samples(test_set, int(len(forget_set)),  args['dataset_name'])
    else:
        unseen_sample, _ = sample_target_samples(shadow_um, int(len(forget_set)), args['dataset_name'])
    for i, idx in enumerate(unseen_sample.indices):
        data, label = unseen_sample.dataset[idx]
        save_path_unseen_sample = f'{save_path}/{i}'
        if not os.path.exists(save_path_unseen_sample):
            os.makedirs(save_path_unseen_sample)
        unseen_sample_original_model_posterior = original_model.predict_proba(data)
        unseen_sample_unlearned_model_posterior = unlearned_model.predict_proba(data)
        torch.save(unseen_sample_original_model_posterior,
                   f"{save_path_unseen_sample}/unseen_sample_original_model_posterior.pth")
        torch.save(unseen_sample_unlearned_model_posterior,
                   f"{save_path_unseen_sample}/unseen_sample_unlearned_model_posterior.pth")
        torch.save(label, f"{save_path_unseen_sample}/unseen_sample_label.pth")

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=args['batch_size'], shuffle=True)
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args['batch_size'], shuffle=True)

    forget_set_acc_original = original_model.test_model_acc(forget_loader)
    retain_set_acc_original = original_model.test_model_acc(retain_loader)
    test_acc_original = original_model.test_model_acc(test_loader)
    forget_set_acc_unlearned = unlearned_model.test_model_acc(forget_loader)
    retain_set_acc_unlearned = unlearned_model.test_model_acc(retain_loader)
    test_acc_unlearned = unlearned_model.test_model_acc(test_loader)

    metrics_dict = {
        "forget_set_acc_original": round(forget_set_acc_original, 4),
        "retain_set_acc_original": round(retain_set_acc_original, 4),
        "test_acc_original": round(test_acc_original, 4),
        "forget_set_acc_unlearned": round(forget_set_acc_unlearned, 4),
        "retain_set_acc_unlearned": round(retain_set_acc_unlearned, 4),
        "test_acc_unlearned": round(test_acc_unlearned, 4),
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)


import torch
from torch.utils.data import Subset
from tqdm import tqdm
import random
import numpy as np


def calculate_confidence_with_subsets(model, train_loader, K, device):

    model.to(device)
    model.eval()
    K = int(K)

    indices = list(range(len(train_loader.dataset)))
    confidences = []
    entropies = []

    inputs = torch.stack([train_loader.dataset[i][0] for i in indices]).to(device)
    outputs = model(inputs)
    probabilities = torch.softmax(outputs, dim=1)

    top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
    confidence = top2_probs[:, 0] - top2_probs[:, 1]
    confidences.extend(confidence.cpu().detach().numpy())

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    entropies.extend(entropy.cpu().detach().numpy())

    sorted_confidence_indices = [idx for _, idx in sorted(zip(confidences, indices), key=lambda x: x[0])]
    sorted_confidence_values = sorted(confidences)

    sorted_entropy_indices = [idx for _, idx in sorted(zip(entropies, indices), key=lambda x: x[0], reverse=True)]
    sorted_entropy_values = sorted(entropies, reverse=True)

    low_confidence_indices = sorted_confidence_indices[:K]
    high_confidence_indices = sorted_confidence_indices[-K:]

    high_entropy_indices = sorted_entropy_indices[:K]
    low_entropy_indices = sorted_entropy_indices[-K:]

    random_indices = random.sample(indices, K)

    all_indices = set(indices)
    remaining_after_high_indices = list(all_indices - set(high_confidence_indices))
    remaining_after_low_indices = list(all_indices - set(low_confidence_indices))
    remaining_after_high_entropy_indices = list(all_indices - set(high_entropy_indices))
    remaining_after_low_entropy_indices = list(all_indices - set(low_entropy_indices))
    remaining_after_random_indices = list(all_indices - set(random_indices))

    high_confidence_subset = Subset(train_loader.dataset, high_confidence_indices)
    remaining_after_high = Subset(train_loader.dataset, remaining_after_high_indices)

    low_confidence_subset = Subset(train_loader.dataset, low_confidence_indices)
    remaining_after_low = Subset(train_loader.dataset, remaining_after_low_indices)

    high_entropy_subset = Subset(train_loader.dataset, high_entropy_indices)
    remaining_after_high_entropy = Subset(train_loader.dataset, remaining_after_high_entropy_indices)

    low_entropy_subset = Subset(train_loader.dataset, low_entropy_indices)
    remaining_after_low_entropy = Subset(train_loader.dataset, remaining_after_low_entropy_indices)

    random_subset = Subset(train_loader.dataset, random_indices)
    remaining_after_random = Subset(train_loader.dataset, remaining_after_random_indices)

    return (
        high_confidence_subset, remaining_after_high,
        low_confidence_subset, remaining_after_low,
        high_entropy_subset, remaining_after_high_entropy,
        low_entropy_subset, remaining_after_low_entropy,
        random_subset,remaining_after_random
    )


class TransformedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform =  transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label

def get_top_bottom_n_indices(lst, n):
    sorted_indices = np.argsort(lst)
    bottom_n_indices = sorted_indices[:n].tolist()
    top_n_indices = sorted_indices[-n:].tolist()
    return top_n_indices, bottom_n_indices


def add_gaussian_noise(image, mean=0, std=0.1):

    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image