# TC-MIA


This repository is the official implementation of the paper: 
#### Revisiting Privacy Leakage in Machine Unlearning: A Tri-class Membership Inference Approach


## Installation

You can install all requirements with:
```bash
pip install -r requirements.txt
```

1. Training target and shadow models.

```bash
python main.py --pre_train both --U_method retrain --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
python main.py --pre_train both --U_method retrain --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:0
```

2. TC_MIA
```bash
python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
```
2.1 U_leak (baseline)

```bash
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
```

2.2 Double attack (baseline)
```bash
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
```