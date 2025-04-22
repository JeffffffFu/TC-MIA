import logging
import joblib
from sympy import false
from torch.utils.data import Subset
from torch.utils.data import TensorDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from opacus import PrivacyEngine
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from model.ResNet import resnet18, resnet50, resnet18_dp
from model.ResNet20 import  resnet182
from model.VGG import vgg11_bn, vgg19_bn, vgg13_bn


class DNN(nn.Module):
    def __init__(self, args=None):
        super(DNN, self).__init__()

        self.logger = logging.getLogger("DNN")
        self.args = args
        self.device = args['device']
        if args['dataset_name']=='tinyimagenet':
            self.imagenet = True
        else:
            self.imagenet = False
        if args['dataset_name']=='cifar100':
            self.num_classes = 100
        elif args['dataset_name']=='celebA':
            self.num_classes = 40
        elif args['dataset_name']=='tinyimagenet':
            self.num_classes = 200
        else:
            self.num_classes = 10
        self.model = self.determine_net(args['net_name'])

    def determine_net(self, net_name, pretrained=False):
        self.logger.debug("determin_net for %s" % net_name)
        self.in_dim = {
            "location": 168,
            "adult": 14,
            "accident": 29,
            "stl10": 96*96*3,
            "cifar10": 32*32*3,
            "cifar100": 32 * 32 * 3,
            "svhn": 32 * 32 * 3,
            "celebA": 128 *128*3,
            "mnist": 28*28*1,
            "fmnist": 28*28*1,
            "cinic10": 32 * 32 * 3,
            "tinyimagenet": 224 * 224 * 3,
        }


        in_dim = self.in_dim[self.args['dataset_name']]
        out_dim = self.num_classes
        imagenet=self.imagenet
        if net_name == "mlp":
            return MLPTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "logistic":
            return LRTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "simple_cnn":
            return Simple_CNN_Tanh(num_classes=out_dim)
        elif net_name == "simple_cnn_dropout":
            return Simple_CNN_Tanh_dropout(num_classes=out_dim)
        elif net_name == "resnet18":
            return resnet18(num_classes=out_dim)
        elif net_name == "resnet18_dp":
            return resnet18_dp(num_classes=out_dim)
        elif net_name == "resnet50":
            return resnet50(num_classes=out_dim)
        elif net_name == "densenet":
            return models.densenet121(num_classes=out_dim)
        elif net_name == "vgg":
            return vgg11_bn(3, out_dim)
        elif net_name == "CNN_MNIST":
            return CNN_MNIST()
        elif net_name == "DT":
            return DecisionTreeClassifier()
        elif net_name == "RF":
            return RandomForestClassifier()
        else:
            raise Exception("invalid net name")

    def train_model(self, train_loader, test_loader, save_name=None):
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'],weight_decay=1e-4)
        if self.args['optim'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=1e-4)


        criterion = nn.CrossEntropyLoss()
        run_result = []

        self.model.train()
        test_acc=0.
        for epoch in range(self.args['num_epochs']):
            losses = []

            for data, target  in train_loader:

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())


            train_acc = self.test_model_acc(train_loader)
           # train_acc = self.test_model_acc(train_loader)

            test_acc = self.test_model_acc(test_loader)
            print(f' epoch:{epoch} | train acc:{round(train_acc, 4)} | test acc: {round(test_acc, 4)}')

        # # self.logger.debug('epoch %s: train acc %s | test acc %s | ovf %s' % (epoch, train_acc, test_acc, train_acc - test_acc))
        # run_result.append([epoch, np.mean(losses), train_acc, test_acc])
        #


    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))

    def predict_proba2(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            feature = test_case[0][0]
            feature = torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()

    def predict_proba(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            feature = torch.unsqueeze(test_case.to(torch.float32), 0).to(self.device)
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()
    # def predict_proba(self, test_case):
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         logits = self.model(test_case)
    #         posterior = F.softmax(logits, dim=1)
    #         return posterior.detach().cpu().numpy()

    def test_model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():

            for data, target  in test_loader:


                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data).to(self.device)

                pred = outputs.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

            return correct / len(test_loader.dataset)

    def logits(self,target_sample):
        if isinstance(target_sample, Subset):
            with torch.no_grad():
                feature=target_sample[0][0]
                label=target_sample[0][1]
                feature =torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
                label=torch.unsqueeze(torch.tensor(label, dtype=torch.long),0).to(self.device)
                logits = self.model(feature).to(self.device)
                logits=logits[0].detach().cpu().numpy()
                probs=np.exp(logits)/np.sum(np.exp(logits))
                confidence=np.max(probs)
        else:
            ValueError("This is not a Subset.")
        return confidence

    def forward_propagation(self, target_sample):
        self.model.eval()
        self.model = self.model.to(self.device)
        return self.model(target_sample)

    def forward(self,x):
        x=self.model(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d( in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x



class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        if x.dim() == 2:  # 如果输入形状是 (batch_size, 10)
            x = x.unsqueeze(1).unsqueeze(3)  # 调整为 (batch_size, 1, 10, 1)
        x=self.conv(x)
        return x

def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class Simple_CNN_Tanh(nn.Module):
    def __init__(self,num_classes=10, in_channels=3, input_norm=None,**kwargs):
        super(Simple_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.num_classes=num_classes

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh
       # act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, self.num_classes))
        else:
            self.classifier = nn.Linear(c * 4 * 4, self.num_classes)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Simple_CNN_Tanh_dropout(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_norm=None, drop_p=0.95, **kwargs):
        super(Simple_CNN_Tanh_dropout, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_p = drop_p

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        c = self.in_channels

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.drop_p > 0:
                    layers += [nn.Dropout2d(self.drop_p)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(
                nn.Linear(c * 4 * 4, hidden),
                act(),
                nn.Dropout(self.drop_p),  # 全连接层后添加Dropout
                nn.Linear(hidden, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.drop_p),  # 输入层Dropout
                nn.Linear(c * 4 * 4, self.num_classes)
            )
    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class MLPTorchNet(nn.Module):
    def __init__(self, in_dim=168, out_dim=9):
        super(MLPTorchNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # temperature = 4
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x


class LRTorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

