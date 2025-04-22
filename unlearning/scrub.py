import copy
import time
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
import torch.nn.functional as F
def scrub(args):
    scrub_save_target_for_population_attack(args)
    scrub_save_shadow_for_population_attack(args)


def scrub_save_target_for_population_attack(args):
    print("target_training----------------")
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
  #  original_model.load_state_dict(torch.load(f"{args['dataset_name']}_original_model.pth", map_location=args['device']))
  #  original_model.to(args['device'])
    original_model.train_model(train_loader, test_loader)

    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        forget_set,retain_set=sample_target_samples(target_m,args['proportion_of_group_unlearn'],args['dataset_name'])

        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False)

        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=scrub_unlearn(unlearned_model, forget_loader, retain_loader, test_loader,args['device'])

        # retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        # forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        # test_acc = unlearned_model.test_model_acc(test_loader)
        # print(f'forget set acc (FA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um, t)


def scrub_save_shadow_for_population_attack(args):
    print("shadow_training----------------")
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

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
        unlearned_model=scrub_unlearn(unlearned_model, forget_loader, retain_loader, test_loader,args['device'])

        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print(f'forget set acc (FA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))
        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um, t)







class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network using KL Divergence."""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss




def scrub_unlearn(original_model,forget_loader,retain_loader,test_loader,device):

        T=1
        scrub_beta=0.0
        scrub_gamma=0.99
        scrub_alpha = 0.1
        smoothing=0.0
        m_steps = 1
        unlearn_epochs=30
        lr=0.001

        teacher = copy.deepcopy(original_model)
        student = copy.deepcopy(original_model)
        model_t = copy.deepcopy(teacher)
        model_s = copy.deepcopy(student)

        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(T)
        criterion_kd = DistillKL(T)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        # optimizer
        optimizer = optim.Adam(trainable_list.parameters(),
                              lr=lr,weight_decay=5e-4)

        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.to(device)
            criterion_list.to(device)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True


        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return (1 - scrub_beta) * averaged_model_parameter + scrub_beta* model_parameter

        swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
        swa_model.to(device)

        for epoch in range(1,unlearn_epochs + 1):

            maximize_loss = 0
            if epoch <= m_steps:
                maximize_loss = train_distill(forget_loader, module_list, swa_model,
                                              criterion_list, optimizer, scrub_gamma,scrub_alpha,scrub_beta, smoothing,"maximize",device,
                                              quiet=False)
            train_acc, train_loss = train_distill(retain_loader, module_list, swa_model, criterion_list,
                                                  optimizer, scrub_gamma,scrub_alpha,scrub_beta, smoothing,
                                                  "minimize", device,quiet=False)


        return model_s


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def train_distill(train_loader, module_list, swa_model, criterion_list, optimizer, scrub_gamma,scrub_alpha,scrub_beta,smoothing, split,device,quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    loss = 0.0

    for idx, data in enumerate(train_loader):
        input, target = data
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)


        # ===================forward=====================
        #feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0
        if split == "minimize":
            loss = scrub_gamma * loss_cls + scrub_alpha * loss_div +scrub_beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, smoothing)

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()

        loss.backward()
        # nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "minimize":

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs, lr_decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = learning_rate
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr, optimizer
