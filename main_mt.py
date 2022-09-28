import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime
import argparse
from utils.misc import *
import shutil
import os
from dataset.dataset import *
from copy import deepcopy
import matplotlib.pyplot as plt
from models.inferenceModel import *
from models.ema import *
from models.resnet import *

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--dataset", type=str, default="cifar", choices=['cifar', 'mnist'])
args_form.add_argument("--max_epoch", type=int, default=8)  # number of opt steps/epochs
args_form.add_argument("--L", type=int, default=256)  # repr dim
args_form.add_argument("--batch_size", type=int, default=64)
args_form.add_argument("--lr", type=float, default=1e-3)
args_form.add_argument("--decay", type=float, default=1e-5)
args_form.add_argument("--ema_decay", type=float, default=0.999)
args_form.add_argument("--nb_pretrain_epochs", type=int, default=32)
args_form.add_argument("--num_labeled", type=int, default=4000)  # 3000, 1500, 900, 600, 300
args_form.add_argument("--mu", type=float, default=1)
args_form.add_argument("--random_hyper", default=False, action="store_true")



def main(seed):
    print(datetime.now())

    args = args_form.parse_args()
    args.baseline = "mt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    args.logs = defaultdict(OrderedDict)
    name = "./results/%s/mt_%s@%d_%d.pytorch" % (args.dataset, args.dataset, args.num_labeled, seed)
    args.t = 0
    args.seed = seed
    set_seed(args)
    print(args)

    # Data, models and initial state
    if args.dataset == 'mnist':
        args.in_dim = (1, 28, 28)
        args.C = 10
        student = InferenceModel(args.in_dim, args.L, args.C).to(device).train()
        val_train_dataset, val_test_dataset, unlabeled_dataset, test_dataset, finetune_dataset = get_mnist(args)
    elif args.dataset == 'cifar':
        args.in_dim = (3, 32, 32)
        args.C = 100
        student = ResNet50(args.C)
        _, _, unlabeled_dataset, _ = get_cifar10(args)
        val_train_dataset, val_test_dataset, _, test_dataset, finetune_dataset = get_cifar100(args)
    else: raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True)
    val_train_loader = torch.utils.data.DataLoader(val_train_dataset, batch_size=args.batch_size, shuffle=False)
    val_test_loader = torch.utils.data.DataLoader(val_test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)

    student_opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.decay)  # , momentum=0.9
    mse = nn.MSELoss()
    pretrain(args, student, finetune_loader, val_test_loader, test_loader)
    teacher = ModelEMA(args, student, args.ema_decay)

    for e in range(args.max_epoch):
        torch.save({"args.logs": args.logs}, name)
        sys.stdout.flush()

        # Encoder training
        student.train()
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            student_opt.zero_grad()

            if batch_idx % 16 == 0 and batch_idx != 0:
                assess(args, teacher.ema, val_train_loader, data='val_train')
                assess(args, teacher.ema, val_test_loader, data='val_test')
                assess(args, teacher.ema, test_loader, data='test')
                console_log(args)

            consistency_loss_unsup = mse(student(images).softmax(1), teacher.ema(images).softmax(1).detach())

            d_i = np.random.choice(len(finetune_dataset), size=args.batch_size, replace=False)
            images, targets = [], []
            for _, x in enumerate(d_i):
                images.append(finetune_dataset[x][0])
                targets.append(finetune_dataset[x][1])
            images, targets = torch.stack(images).to(device), torch.tensor(targets, device=device)
            # images, targets = val_train_dataset[d_i][0].to(device), val_train_dataset[d_i][1].to(device)

            consistency_loss_sup = mse(student(images).softmax(1), teacher.ema(images).softmax(1).detach())
            loss_sup = xent(student(images).softmax(1), targets)

            loss = (loss_sup + args.mu / 2 * (consistency_loss_unsup + consistency_loss_sup))
            loss.backward()
            student_opt.step()

            teacher.update(student)

            args.logs['train_loss'][args.t] = loss.item()
            args.t += 1
            teacher.decay = min(1 - 1. / args.t, teacher.decay)

    torch.save({"args.logs": args.logs}, name)

    assess(args, teacher.ema, val_train_loader, data='val_train')
    assess(args, teacher.ema, val_test_loader, data='val_test')
    assess(args, teacher.ema, test_loader, data='test')
    console_log(args, main_train=False)


def pretrain(args, model, val_train_loader, val_test_loader, test_loader):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for _ in range(args.nb_pretrain_epochs):
        assess(args, model, val_train_loader, data='val_train')
        assess(args, model, val_test_loader, data='val_test')
        assess(args, model, test_loader, data='test')
        console_log(args, main_train=False)

        for batch_idx, (images, labels) in enumerate(val_train_loader):
            opt.zero_grad()

            images, labels = images.to(device), labels.to(device)

            distr = model(images).softmax(1)
            loss = xent(distr, labels)
            loss.backward()

            opt.step()

        args.t += 1

    return 0


def assess(args, model, data_loader, data):
    model.eval()
    with torch.no_grad():
        loss = 0.
        correct = 0.
        total = 0.
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            distr = model(images).softmax(1)
            _, pred = distr.max(1)

            loss += xent(distr, labels, reduction='sum').item()
            correct += pred.eq(labels).sum().item()
            total += images.size(0)
        if data == 'val_train':
            args.logs['val_train_loss'][args.t] = loss / total
            args.logs['val_train_acc'][args.t] = 100. * correct / total
        if data == 'val_test':
            args.logs['val_test_loss'][args.t] = loss / total
            args.logs['val_test_acc'][args.t] = 100. * correct / total
        if data == 'test':
            args.logs['test_loss'][args.t] = loss / total
            args.logs['test_acc'][args.t] = 100. * correct / total

    model.train()
    return 0

if __name__ == '__main__':
    for _, seed in enumerate([0,1,2,3,4]):
        main(seed)