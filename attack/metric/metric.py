import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = max(tpr[fpr==fpr[ind]])
    return tpr_res

def all_metrics(y_target,probs):



    auc_target = metrics.roc_auc_score(y_target, probs)
    tpr_001fpr = tpr_at_fpr(y_target, probs, 0.01)
    tpr_0005fpr = tpr_at_fpr(y_target, probs, 0.005)
    tpr_005fpr = tpr_at_fpr(y_target, probs, 0.05)
    all_metrics_value=[ auc_target,tpr_001fpr, tpr_0005fpr, tpr_005fpr]
   # print(f'Accuracy: {acc_target:.3f}, Precision: {prec_target:.3f}, Recall: {recall_target:.3f}, F1-Score: {f1_target:.3f}, AUC: {auc_target:.3f}, TPR at 0.01 FPR: {tpr_001fpr:.3f}, TPR at 0.10 FPR: {tpr_10fpr}')
    return all_metrics_value


def all_metrics_for_three_class(y_target, y_pred,probs):
    # transform 3 classes to 2 classes

    ACC_list=[]
    PRE_list=[]
    REC_list=[]
    F1_list=[]
    AUC_list=[]
    TPR001_list=[]
    TPR0005_list=[]
    TPR005_list=[]
    y_target = np.array(y_target)
    y_pred= np.array(y_pred)

    prob0 = probs[y_target == 0]
    prob1 = probs[y_target == 1]
    prob2 = probs[y_target == 2]
    count_dict = Counter(y_pred)

    for target_class in range(3):

        acc = np.mean(y_pred[y_target == target_class] == target_class)
        tp = np.sum((y_pred == target_class) & (y_target == target_class))
        fp = np.sum((y_pred == target_class) & (y_target != target_class))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        fn = np.sum((y_pred != target_class) & (y_target == target_class))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ACC_list.append(round(acc,4))
        PRE_list.append(round(precision,4))
        REC_list.append(round(recall,4))
        F1_list.append(round(f1,4))

        if target_class == 0:
            prob_main = prob0
            prob_other = np.concatenate([prob1[:len(prob1) // 2], prob2[:len(prob2) // 2]])
            new_probs = np.concatenate([prob_main, prob_other])
            probs_pred=new_probs[:,target_class]
        elif target_class == 1:
            prob_main = prob1
            prob_other = np.concatenate([prob0[:len(prob0) // 2], prob2[:len(prob2) // 2]])
            new_probs = np.concatenate([prob_main, prob_other])
            probs_pred=new_probs[:,target_class]

        else:
            prob_main = prob2
            prob_other = np.concatenate([prob0[:len(prob0) // 2], prob1[:len(prob1) // 2]])
            new_probs = np.concatenate([prob_main, prob_other])
            probs_pred=new_probs[:,target_class]

        new_targets = np.array([1] * len(prob_main) + [0] * len(prob_other))
        all_metrics_value=all_metrics(new_targets,probs_pred)
      #  y_target_binary = [1 if y == target_class else 0 for y in y_target]
      #  all_metrics_value=all_metrics(y_target_binary,probs_pred)
        AUC_list.append(round(all_metrics_value[0],4))
        TPR001_list.append(round(all_metrics_value[1],4))
        TPR0005_list.append(round(all_metrics_value[2],4))
        TPR005_list.append(round(all_metrics_value[3],4))
    micro_F1=f1_score(y_target, y_pred, average='micro')
    return micro_F1,ACC_list,PRE_list,REC_list,F1_list,AUC_list,TPR001_list,TPR0005_list,TPR005_list

def save_metric(Micro_F1_list,ACC_list_list,PRE_list_List,REC_list_list,F1_list_list,AUC_list_list,TPR001_list_list,TPR0005_list_list,TPR005_list_list,args):
   # best_AUC_list = max(AUC_list_list, key=lambda x: sum(x) / len(x))
    best_Micro_F1 = max(Micro_F1_list)
    index=Micro_F1_list.index(best_Micro_F1)
    best_ACC_list = ACC_list_list[index]
    best_PRE_list = PRE_list_List[index]
    best_REC_list = REC_list_list[index]
    best_F1_list = F1_list_list[index]
    best_AUC_list = AUC_list_list[index]
    best_TPR001_list = TPR001_list_list[index]
    best_TPR0005_list = TPR0005_list_list[index]
    best_TPR005_list = TPR005_list_list[index]

    best_ACC=np.mean(best_ACC_list)
    best_PRE=np.mean(best_PRE_list)
    best_REC=np.mean(best_REC_list)
    best_F1=np.mean(best_F1_list)
    best_AUC=np.mean(best_AUC_list)

    best_ACC_str = "{:.2f}%".format(best_ACC * 100)
    best_PRE_str = "{:.2f}%".format(best_PRE * 100)
    best_REC_str = "{:.2f}%".format(best_REC * 100)
    best_Micro_F1 = "{:.2f}%".format(best_Micro_F1 * 100)
    best_F1_str = "{:.2f}%".format(best_F1 * 100)
    best_AUC_str = "{:.2f}%".format(best_AUC * 100)

    best_ACC_per_str = "/".join(["{:.2f}%".format(x * 100) for x in best_ACC_list])
    best_PRE_per_str = "/".join(["{:.2f}%".format(x * 100) for x in best_PRE_list])
    best_REC_per_str = "/".join(["{:.2f}%".format(x * 100) for x in best_REC_list])
    best_F1_per_str = "/".join(["{:.2f}%".format(x * 100) for x in best_F1_list])
    best_AUC_per_str = "/".join(["{:.2f}%".format(x * 100) for x in best_AUC_list])
    best_TPR001_str = "/".join(["{:.2f}%".format(x * 100) for x in best_TPR001_list])
    best_TPR0005_str = "/".join(["{:.2f}%".format(x * 100) for x in best_TPR0005_list])
    best_TPR005_str = "/".join(["{:.2f}%".format(x * 100) for x in best_TPR005_list])

    metrics_dict = {
        "Best ACC": best_ACC_str,
        "Best ACC_per": best_ACC_per_str,
        "Best PRE": best_PRE_str,
        "Best PRE_per": best_PRE_per_str,
        "Best REC": best_REC_str,
        "Best REC_per": best_REC_per_str,
        "Best Micro F1": best_Micro_F1,
        "Best Macro F1": best_F1_str,
        "Best F1_per": best_F1_per_str,
        "Best AUC": best_AUC_str,
        "Best AUC_per": best_AUC_per_str,
        "Best TPR001": best_TPR001_str,
        "Best TPR0005": best_TPR0005_str,
        "Best TPR005": best_TPR005_str,
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    save_path = os.getcwd() + f"/result/{args['attack_method']}/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/"

    os.makedirs(save_path, exist_ok=True)

    data_to_save.to_csv(f"{save_path}/attack_acc.csv", index=False)

def acc_for_three_domain(y_unseen, y_retain,y_forgot, p_unseen, p_retain,p_forgot):
    acc_unseen=sum(p == t for p, t in zip(y_unseen, p_unseen)) / len(y_unseen)
    acc_retain=sum(p == t for p, t in zip(y_retain, p_retain)) / len(y_retain)
    acc_forgot=sum(p == t for p, t in zip(y_forgot, p_forgot)) / len(y_forgot)

    y_pred=p_unseen+p_retain+p_forgot
    y_true=y_unseen+y_retain+y_forgot
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f'accuracy: {accuracy:.3f},acc_unseen: {acc_unseen:.3f},acc_retain: {acc_retain:.3f},acc_forgot: {acc_forgot:.3f}')
    return accuracy