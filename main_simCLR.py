import torch
import torch.nn as nn
import torchvision.utils
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime
import argparse
import shutil
import os
from torch.utils.data import DataLoader
from dataset.dataset_simCLR import *
from models.inferenceModel import *
from models.resnet import ResNet50
from utils.misc import *
from copy import deepcopy
import matplotlib.pyplot as plt

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "mnist"])
args_form.add_argument("--max_epoch", type=int, default=4)  # number of opt steps/epochs
args_form.add_argument("--L", type=int, default=256)  # repr dim
args_form.add_argument("--batch_size", type=int, default=64)
args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--decay", type=float, default=1e-5)
args_form.add_argument("--mlp_lr", type=float, default=1e-2)
args_form.add_argument("--mlp_decay", type=float, default=1e-5)
args_form.add_argument("--pretrain_lr", type=float, default=1e-2)
args_form.add_argument("--nb_pretrain_epochs", type=int, default=32)
args_form.add_argument("--nb_finetune_epochs", type=int, default=32)
args_form.add_argument("--num_labeled", type=int, default=1800)  # 3000, 1500, 900, 600, 300
args_form.add_argument("--tau", type=float, default=0.1)  # 0.05, 0.1, 0.5, 1.0
args_form.add_argument("--mu", type=float, default=1)


def main(seed):
    print(datetime.now())

    args = args_form.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = seed
    set_seed(args)
    args.logs = defaultdict(OrderedDict)
    print(args)
    name = "./results/%s/simclr_%s@%d_%d.pytorch" % (args.dataset, args.dataset, args.num_labeled, seed)
    args.t = 0


    # Data
    train_data, val_train_data, val_test_data, test_data = get_data(args)

    train_loader = DataLoader(train_data, batch_size=args.batch_size >> 1, shuffle=True)
    val_train_loader = DataLoader(val_train_data, batch_size=args.batch_size, shuffle=True)
    val_test_loader = DataLoader(val_test_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    val_train_iter = iter(val_train_loader)


    # Models and initial state
    if args.dataset == 'mnist':
        enc = TargetModel(args.in_dim, args.L).to(device).train()
    elif args.dataset == 'cifar':
        enc = ResNet50(args.L).to(device).train()
    enc_opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.decay)  # , momentum=0.9

    head = ClassifierModel(args.L, args.L).to(device).train()
    head_opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.decay)
    mlp = ClassifierModel(args.L, args.num_classes).to(device).train()
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_decay)

    cos_dist = nn.CosineSimilarity()

    train(args, enc, mlp, val_train_loader, val_test_loader, test_loader, purpose='pretrain')

    for e in range(args.max_epoch):

        torch.save({"args.logs": args.logs}, name)
        sys.stdout.flush()

        # Encoder training
        enc.train()
        for batch_idx, ((raw_images, _), (aug1_images, _), (aug2_images, _)) in enumerate(train_loader):
            aug1_images, aug2_images = aug1_images.to(device), aug2_images.to(device)
            enc_opt.zero_grad()
            head_opt.zero_grad()
            mlp_opt.zero_grad()

            if batch_idx%16 == 0 and batch_idx != 0:
                assess_linlayer(args, enc, mlp, val_train_loader, data="val_train")
                assess_linlayer(args, enc, mlp, val_test_loader, data="val_test")
                assess_linlayer(args, enc, mlp, test_loader, data="test")
                console_log(args)

            a = head(enc(aug1_images))
            a_norm = torch.norm(a, dim=1).reshape(-1, 1).to(device)
            a_cap = torch.div(a, a_norm).to(device)

            b = head(enc(aug2_images))
            b_norm = torch.norm(b, dim=1).reshape(-1, 1).to(device)
            b_cap = torch.div(b, b_norm).to(device)

            a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0).to(device)
            b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0).to(device)

            sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t())
            exp_sim_by_tau = torch.exp(torch.div(sim, args.tau))
            mask = torch.ones_like(exp_sim_by_tau, device=device) - torch.eye(exp_sim_by_tau.shape[0], exp_sim_by_tau.shape[1], device=device)
            exp_sim_by_tau = mask * exp_sim_by_tau

            numerators = torch.exp(torch.div(cos_dist(a_cap_b_cap, b_cap_a_cap), args.tau))
            denominators = torch.sum(exp_sim_by_tau, dim=1)
            num_by_den = torch.div(numerators, denominators)
            neglog_num_by_den = -torch.log(num_by_den)
            unsup_loss = neglog_num_by_den.mean()

            # Sup. part
            try:
                images, labels = val_train_iter.next()
                assert images.shape[0] == args.batch_size
            except:
                val_train_iter = iter(val_train_loader)
                images, labels = val_train_iter.next()
            images, labels = images.to(device), labels.to(device)

            distr = mlp(enc(images)).softmax(1)
            sup_loss = xent(distr, labels)

            (unsup_loss + args.mu * sup_loss).backward()
            enc_opt.step()
            mlp_opt.step()
            head_opt.step()

            args.logs['train_loss'][args.t] = unsup_loss.item()
            args.t += 1  # nb of opt. steps taken
            # args.t += args.batch_size  # nb of images seen

    train(args, enc, mlp, val_train_loader, val_test_loader, test_loader, purpose='finetune')
    assess_linlayer(args, enc, mlp, val_train_loader, data="val_train")
    assess_linlayer(args, enc, mlp, val_test_loader, data="val_test")
    assess_linlayer(args, enc, mlp, test_loader, data="test")
    console_log(args, main_train=False)
    torch.save({"args.logs": args.logs}, name)


def train(args, enc, mlp, val_train_loader, val_test_loader, test_loader, purpose):
    lr = args.lr if purpose=='pretrain' else args.lr / 100
    nb_epochs = args.nb_pretrain_epochs if purpose=='pretrain' else args.nb_finetune_epochs

    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_decay)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=args.decay)

    enc.train()
    mlp.train()
    for _ in range(nb_epochs):
        for batch_idx, (raw_images, raw_labels) in enumerate(val_train_loader):
            opt.zero_grad()
            mlp_opt.zero_grad()

            images, labels = [], []
            for i in range(raw_images.size(0)):
                images.append(raw_images[i])
                labels.append(raw_labels[i])
            images, labels = torch.stack(images).to(device), torch.stack(labels).to(device)

            distr = mlp(enc(images)).softmax(1)
            loss = xent(distr, labels)

            loss.backward()
            opt.step()
            mlp_opt.step()

        args.t += 1

        assess_linlayer(args, enc, mlp, val_train_loader, data="val_train")
        assess_linlayer(args, enc, mlp, val_test_loader, data="val_test")
        assess_linlayer(args, enc, mlp, test_loader, data="test")
        console_log(args, main_train=False)

    enc.train()
    mlp.train()
    return 0


def assess_linlayer(args, enc, mlp, data_loader, data):
    enc.eval()
    mlp.eval()
    with torch.no_grad():
        loss = 0.
        correct = 0.
        total = 0.
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            _, pred = mlp(enc(images)).softmax(1).max(1)
            loss += xent(mlp(enc(images)).softmax(1), labels, reduction='sum').item()
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    if data == 'val_train':
        args.logs['val_train_loss'][args.t] = loss / total
        args.logs['val_train_acc'][args.t] = 100. * correct / total
    if data == 'val_test':
        args.logs['val_test_loss'][args.t] = loss / total
        args.logs['val_test_acc'][args.t] = 100. * correct / total
    if data == 'test':
        args.logs['test_loss'][args.t] = loss / total
        args.logs['test_acc'][args.t] = 100. * correct / total

    enc.train()
    mlp.train()
    return 0


if __name__ == '__main__':
    for _, seed in enumerate([0, 1, 2, 3, 4]):
        main(seed)


