#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch
from torch.utils.data import Subset


def construct_dataset(imgs, labels, args): # todo: move to another script? like utils?
    target_dataset, target_labels, shadow_dataset, shadow_labels = train_test_split(imgs, labels, test_size=0.5, random_state=args['random'])
    target_m, target_m_labels, target_um, target_um_labels = train_test_split(target_dataset, target_labels, test_size=0.2, random_state=args['random'])
    shadow_m, shadow_m_labels, shadow_um, shadow_um_labels = train_test_split(shadow_dataset, shadow_labels, test_size=0.2, random_state=args['random'])

    # save the datasets
    data_split_loc = f"data/splits/{args['dataset_name']}"
    if not os.Path.exists(data_split_loc):
        os.mkdir(data_split_loc)
    np.save(f"{data_split_loc}/target_m.npy", target_m)
    np.save(f"{data_split_loc}/target_m_labels.npy", target_m_labels)
    np.save(f"{data_split_loc}/target_um.npy", target_um)
    np.save(f"{data_split_loc}/target_um_labels.npy", target_um_labels)
    np.save(f"{data_split_loc}/shadow_m.npy", shadow_m)
    np.save(f"{data_split_loc}/shadow_m_labels.npy", shadow_m_labels)
    np.save(f"{data_split_loc}/shadow_um.npy", shadow_um)
    np.save(f"{data_split_loc}/shadow_um_labels.npy", shadow_um_labels)

    return target_m, target_m_labels, target_um, target_um_labels, shadow_m, shadow_m_labels, shadow_um, shadow_um_labels


def split_dataset(train_dataset, random):


    target_dataset, shadow_dataset = train_test_split(train_dataset, test_size=0.5, random_state=random)
   # target_m, target_um= train_test_split(target_dataset, test_size=0.2, random_state=random)
    shadow_m, shadow_um = train_test_split(shadow_dataset, test_size=0.2, random_state=random)


    return target_dataset, shadow_m, shadow_um

# for impact of the type of removed instances
def split_dataset2(train_dataset, random):


    target_dataset, shadow_dataset = train_test_split(train_dataset, test_size=0.95, random_state=random)
    target_m, target_um = train_test_split(target_dataset, test_size=0.5, random_state=random)
    shadow_m, shadow_um = train_test_split(shadow_dataset, test_size=0.2, random_state=random)


    return target_m,target_um,  shadow_m, shadow_um

#keep transform or augmention after split
def split_dataset3(train_dataset, random):

    indices = list(range(len(train_dataset)))
    target_indices, shadow_indices = train_test_split(indices, test_size=0.5, random_state=random, shuffle=False)

    target_dataset = Subset(train_dataset, target_indices)
    shadow_dataset = Subset(train_dataset, shadow_indices)

   # target_dataset, shadow_dataset = train_test_split(train_dataset, test_size=0.0001, random_state=random, shuffle=False)


    return target_dataset, shadow_dataset