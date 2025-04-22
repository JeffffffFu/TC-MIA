import os

import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def attack_feature_base(P_in, P_out, label_list,method):
    assert method in ["CDS", "CT",'DC','SC','DD','SD','ED',"CP", "DF", 'SM','test','LO']


    attack_feature=[]
    if  method == "CT":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            attack_feature.append(
                [posterior_in[label], posterior_out[label]])

    elif method == "CDS":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            feature_TL_sum=posterior_in[label]+posterior_out[label]

            attack_feature.append(
                [feature_TL_diff,feature_TL_sum])

    elif method == "DF":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_diff])

    elif method == "SM":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_sum=posterior_in[label]+posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_sum])

    elif method == "test":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            TL_original=posterior_in[label]
            TL_unlearned=posterior_out[label]
            if TL_original<0.5:
                TL_unlearned=0.1
            attack_feature.append(
                [TL_original, TL_unlearned])

    elif method == "DC" or method == "CP":

        return np.concatenate([P_in, P_out], axis=1)

    elif method == "SC":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return np.concatenate([P_in, P_out], axis=1)

    elif method == "DD":
        return P_in - P_out

    elif method == "SD":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return P_in - P_out

    elif method == "ED":
        return np.sqrt((P_in - P_out) ** 2)

    elif method == "LO":  #publishing label only

        for posterior_in, posterior_out, label in zip(P_in, P_out, label_list):
            # Normalize posterior_in
            max_index_in = np.argmax(posterior_in)
            posterior_in = np.zeros_like(posterior_in)
            posterior_in[max_index_in] = 1

            # Normalize posterior_out
            max_index_out = np.argmax(posterior_out)
            posterior_out = np.zeros_like(posterior_out)
            posterior_out[max_index_out] = 1

            # Extract the value corresponding to the given label
            attack_feature.append([posterior_in[label], posterior_out[label]])
    else:
        raise ValueError("this algorithm is not exist")

    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)

    return attack_feature

def baseline_prep(args):

    P_original_model_target = []
    P_unlearned_model_target = []
    target_class = []
    P_original_model_shadow = []
    P_unlearned_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]

    target_save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/0"
    target_shadow_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/shadow/0"


    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    # target
    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/{i}"

            # unlearned_sample_posterior
            unlearned_sample_original_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_original_model_posterior.pth")
            unlearned_sample_unlearned_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_unlearned_model_posterior.pth")
            unlearned_sample_label=torch.load(f"{save_path}/target_sample_label.pth")

            P_original_model_target.append(unlearned_sample_original_model_posterior)
            P_unlearned_model_target.append(unlearned_sample_unlearned_model_posterior)
            target_class.append(1)
            target_sample_label.append(unlearned_sample_label)

            # unseen_sample_posterior
            unseen_sample_original_model_posterior = torch.load(f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior = torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label=torch.load(f"{save_path}/unseen_sample_label.pth")

            P_original_model_target.append(unseen_sample_original_model_posterior)
            P_unlearned_model_target.append(unseen_sample_unlearned_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior = torch.load(f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior = torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label=torch.load(f"{save_path}/retain_sample_label.pth")

            P_original_model_target.append(retain_sample_original_model_posterior)
            P_unlearned_model_target.append(retain_sample_unlearned_model_posterior)
            target_class.append(2)
            target_sample_label.append(retain_sample_label)

    #shadow
    for t in tqdm(range(args['observations']), desc='preparing members'):
        for i in range(shadow_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/shadow/{t}/{i}"

            #unlearned_sample_posterior
            unlearned_sample_original_model_posterior = torch.load(f"{save_path}/unlearned_sample_original_model_posterior.pth")
            unlearned_sample_unlearned_model_posterior = torch.load(f"{save_path}/unlearned_sample_unlearned_model_posterior.pth")
            unlearned_sample_label=torch.load(f"{save_path}/target_sample_label.pth")
            P_original_model_shadow.append(unlearned_sample_original_model_posterior)
            P_unlearned_model_shadow.append(unlearned_sample_unlearned_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(unlearned_sample_label)

            # unseen_sample_posterior
            unseen_sample_original_model_posterior= torch.load(f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior= torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label=torch.load(f"{save_path}/unseen_sample_label.pth")
            P_original_model_shadow.append(unseen_sample_original_model_posterior)
            P_unlearned_model_shadow.append(unseen_sample_unlearned_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior= torch.load(f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior= torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label=torch.load(f"{save_path}/retain_sample_label.pth")
            P_original_model_shadow.append(retain_sample_original_model_posterior)
            P_unlearned_model_shadow.append(retain_sample_unlearned_model_posterior)
            shadow_class.append(2)
            shadow_sample_label.append(retain_sample_label)


    P_original_model_shadow = np.array(P_original_model_shadow)
    P_unlearned_model_shadow = np.array(P_unlearned_model_shadow)
    P_original_model_target = np.array(P_original_model_target)
    P_unlearned_model_target = np.array(P_unlearned_model_target)
    return P_original_model_shadow, P_unlearned_model_shadow, P_original_model_target, P_unlearned_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label

# shadow just have member and non-member
def baseline_prep_for_double_attack(args):

    P_original_model_target = []
    P_unlearned_model_target = []
    target_class = []
    P_original_model_shadow = []
    P_unlearned_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]



    target_save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/0"
    target_shadow_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/shadow/0"

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/{i}"

            unlearned_sample_original_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_original_model_posterior.pth")
            unlearned_sample_unlearned_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_unlearned_model_posterior.pth")
            unlearned_sample_label=torch.load(f"{save_path}/target_sample_label.pth")

            P_original_model_target.append(unlearned_sample_original_model_posterior)
            P_unlearned_model_target.append(unlearned_sample_unlearned_model_posterior)
            target_class.append(1)
            target_sample_label.append(unlearned_sample_label)

            # unseen_sample_posterior
            unseen_sample_original_model_posterior = torch.load(f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior = torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label=torch.load(f"{save_path}/unseen_sample_label.pth")

            P_original_model_target.append(unseen_sample_original_model_posterior)
            P_unlearned_model_target.append(unseen_sample_unlearned_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior = torch.load(f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior = torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label=torch.load(f"{save_path}/retain_sample_label.pth")

            P_original_model_target.append(retain_sample_original_model_posterior)
            P_unlearned_model_target.append(retain_sample_unlearned_model_posterior)
            target_class.append(2)
            target_sample_label.append(retain_sample_label)

    #shadow
    for t in tqdm(range(args['observations']), desc='preparing members'):
        for i in range(shadow_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/shadow/{t}/{i}"
         #   save_path = os.getcwd() + f"/save/retrain/{args['net_name']}/{args['dataset_name']}/0.02/shadow/{t}/{i}"


            # unseen_sample_posterior
            unseen_sample_original_model_posterior= torch.load(f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior= torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label=torch.load(f"{save_path}/unseen_sample_label.pth")
            P_original_model_shadow.append(unseen_sample_original_model_posterior)
            P_unlearned_model_shadow.append(unseen_sample_unlearned_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior= torch.load(f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior= torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label=torch.load(f"{save_path}/retain_sample_label.pth")
            P_original_model_shadow.append(retain_sample_original_model_posterior)
            P_unlearned_model_shadow.append(retain_sample_unlearned_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(retain_sample_label)


    P_original_model_shadow = np.array(P_original_model_shadow)
    P_unlearned_model_shadow = np.array(P_unlearned_model_shadow)
    P_original_model_target = np.array(P_original_model_target)
    P_unlearned_model_target = np.array(P_unlearned_model_target)
    return P_original_model_shadow, P_unlearned_model_shadow, P_original_model_target, P_unlearned_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label


def adjust_ratio_samples(list_a, list_b, list_c, list_d,target_ratio=(2, 1, 2)):

    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    arr_c = np.array(list_c)
    arr_d = np.array(list_d)

    assert len(arr_a) == len(arr_b)==len(arr_c)==len(arr_d)

    unseen_idx = np.where(arr_a == 0)[0]
    forget_idx = np.where(arr_a == 1)[0]
    retain_idx = np.where(arr_a == 2)[0]

    ratio_unseen,ratio_forget,ratio_retain  = target_ratio

    min_count = min(
        len(forget_idx) // ratio_forget,
        len(unseen_idx) // ratio_unseen,
        len(retain_idx) // ratio_retain
    )

    target_forget = min_count * ratio_forget
    target_unseen = min_count * ratio_unseen
    target_retain = min_count * ratio_retain

    np.random.seed(0)

    selected_forget = np.random.choice(forget_idx, target_forget, replace=False)
    selected_unseen = np.random.choice(unseen_idx, target_unseen, replace=False)
    selected_retain = np.random.choice(retain_idx, target_retain, replace=False)


    selected_idx = np.concatenate([selected_forget, selected_unseen, selected_retain])
    selected_idx.sort()

    new_list_a = arr_a[selected_idx]
    new_list_b = arr_b[selected_idx]
    new_list_c = arr_c[selected_idx]
    new_list_d = arr_d[selected_idx]

    return new_list_a.tolist(), new_list_b, new_list_c,new_list_d
