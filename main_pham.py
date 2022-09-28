import argparse
import time

import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from models.resnet import ResNet50
from models.inferenceModel import build_InferenceModel
from dataset.dataset import *
from models.ema import ModelEMA
from utils import *
from collections import defaultdict, OrderedDict
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', type=str, choices=['cifar', 'mnist'], help='dataset name')
parser.add_argument('--num_labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument('--total_steps', default=2**12, type=int, help='number of total steps to run')
parser.add_argument('--eval_step', default=16, type=int, help='number of eval steps to run')
parser.add_argument('--start_step', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--teacher_lr', default=0.05, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.05, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='train weight decay')
parser.add_argument('--ema', default=0.995, type=float, help='EMA decay rate')
parser.add_argument('--use_ema', action='store_true', default=False, help='use ema')
parser.add_argument('--grad_clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true', help='only finetune model on labeled dataset')
parser.add_argument('--finetune_epochs', default=128, type=int, help='finetune epochs')
parser.add_argument('--finetune_batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune_lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune_weight_decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune_momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.6, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=0.7, type=float, help='pseudo label temperature')
parser.add_argument('--lambda_u', default=8, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda_steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--L", type=int, default=256)
parser.add_argument("--nu", type=float, default=1)


def main(seed):
    args = parser.parse_args()
    args.baseline = "pham"
    args.seed = seed
    args.out = 'results/%s/pham_%s@%d_%d' % (args.dataset, args.dataset, args.num_labeled, args.seed)
    args.beginning = datetime.now()
    args.t = 0

    args.gpu = 0

    args.device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else 'cpu'

    args.logs = defaultdict(OrderedDict)
    set_seed(args)

    if args.dataset == 'cifar':
        args.num_classes = 100
        args.in_dim = (3,32,32)
        teacher_model = ResNet50(args.num_classes)
        student_model = ResNet50(args.num_classes)
        val_train_dataset, val_test_dataset, _ , test_dataset, finetune_dataset = get_cifar100(args)
        _, _, train_unlabeled_dataset, _, _ = get_cifar10(args)
    elif args.dataset == 'mnist':
        args.num_classes = 10
        val_train_dataset, val_test_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset = get_mnist(args)
        args.in_dim = (1,28,28)
        teacher_model = build_InferenceModel(args)
        student_model = build_InferenceModel(args)
    else: raise NotImplementedError

    teacher_model.to(args.device)
    student_model.to(args.device)

    val_train_loader = DataLoader(
        val_train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers)

    finetune_loader = DataLoader(
        finetune_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)

    val_test_loader = DataLoader(
        val_test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers)

    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        shuffle=True,
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers)

    avg_student_model = None
    if args.use_ema:
        avg_student_model = ModelEMA(student_model, args.ema, args.weight_decay)

    criterion = create_loss_fn(args)

    t_optimizer = optim.SGD(teacher_model.parameters(), lr=args.teacher_lr, momentum=args.momentum, nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_model.parameters(), lr=args.student_lr, momentum=args.momentum, nesterov=args.nesterov)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, val_train_loader, train_unlabeled_loader, test_loader, val_test_loader, finetune_dataset, finetune_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scaler, s_scaler)
    return


def train_loop(args, val_train_loader, train_loader, test_loader, val_test_loader, finetune_dataset, finetune_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scaler, s_scaler):

    labeled_iter = iter(finetune_loader)
    unlabeled_iter = iter(train_loader)

    for step in range(args.start_step, args.total_steps):
        teacher_model.pretrain()
        student_model.pretrain()

        try:
            images_l, targets = labeled_iter.next()
        except:
            labeled_iter = iter(finetune_loader)
            images_l, targets = labeled_iter.next()

        try:
            (images_uw, images_us), _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(train_loader)
            (images_uw, images_us), _ = unlabeled_iter.next()

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask)
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        if args.use_ema:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            dot_product = s_loss_l_new - s_loss_l_old
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)

            t_loss = t_loss_uda + args.nu * t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()

        teacher_model.zero_grad()
        student_model.zero_grad()

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:

            test_model = avg_student_model if args.use_ema else student_model
            val_train_loss, val_train_acc, top5,  = evaluate(args, val_train_loader, test_model, criterion)
            val_test_loss, val_test_acc, top5,  = evaluate(args, val_test_loader, test_model, criterion)
            test_loss, test_acc, top5,  = evaluate(args, test_loader, test_model, criterion)

            args.logs['test_loss'][args.t] = test_loss
            args.logs['test_acc'][args.t] = test_acc
            args.logs['val_train_loss'][args.t] = val_train_loss
            args.logs['val_train_acc'][args.t] = val_train_acc
            args.logs['val_test_loss'][args.t] = val_test_loss
            args.logs['val_test_acc'][args.t] = val_test_acc
            console_log(args, main_train=False)

            name = "./%s.pytorch" % (str(args.out))
            torch.save({"args.logs": args.logs}, name)

        args.t += 1

    del t_scaler, t_optimizer, teacher_model, train_loader
    del s_scaler, s_optimizer

    finetune(args, finetune_dataset, test_loader, val_train_loader, val_test_loader, student_model, criterion)
    return


def finetune(args, finetune_dataset, test_loader, val_train_loader, val_test_loader, model, criterion):
    model.drop = nn.Identity()
    labeled_loader = DataLoader(
        finetune_dataset,
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=args.finetune_momentum,
        weight_decay=args.finetune_weight_decay,
        nesterov=True)
    scaler = amp.GradScaler(enabled=args.amp)

    for epoch in range(args.finetune_epochs):
        data_time = AverageMeter()
        model.pretrain()
        end = time.time()
        for step, (images, targets) in enumerate(labeled_loader):
            data_time.update(time.time() - end)
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        args.t += 1
        val_train_loss, val_train_acc, top5 = evaluate(args, val_train_loader, model, criterion)
        val_test_loss, val_test_acc, top5 = evaluate(args, val_test_loader, model, criterion)
        test_loss, test_acc, top5 = evaluate(args, test_loader, model, criterion)

        args.logs['val_train_loss'][args.t] = val_train_loss
        args.logs['val_train_acc'][args.t] = val_train_acc
        args.logs['val_test_loss'][args.t] = val_test_loss
        args.logs['val_test_acc'][args.t] = val_test_acc
        args.logs['test_loss'][args.t] = test_loss
        args.logs['test_acc'][args.t] = test_acc
        console_log(args, main_train=False)

        name = "./%s.pytorch" % (str(args.out))
        torch.save({"args.logs": args.logs}, name)

    return


def evaluate(args, data_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy_pham(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

        model.pretrain()
        return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    for _, seed in enumerate([0,1,2,3,4]):
        main(seed)
