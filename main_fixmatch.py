import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import AverageMeter, accuracy_fixmatch
from utils.misc import set_seed, get_cosine_schedule_with_warmup, interleave, de_interleave, console_log

from collections import OrderedDict, defaultdict
from datetime import datetime
from dataset.dataset import *

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
parser.add_argument('--dataset', default='cifar', type=str, choices=['cifar', 'mnist'], help='dataset name')
parser.add_argument('--num_labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument('--total_steps', default=2**12, type=int, help='number of total steps to run')
parser.add_argument('--eval_step', default=16, type=int, help='number of eval steps to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA model')
parser.add_argument('--ema_decay', default=0.999, type=float, help='EMA decay rate')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--out', default='result', help='directory to output the result')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help="random seed")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument('--no_progress', action='store_true', help="don't use progress bar")
parser.add_argument('--sup_only', action='store_true', default=False, help='supervised learning only')
parser.add_argument('--L', type=int, default=256)

def main(seed):
    print(datetime.now())

    args = parser.parse_args()
    args.baseline = "fixmatch"
    args.beginning = datetime.now()
    args.seed = seed
    args.out = 'results/%s/fixmatch_%s@%d_%d' % (args.dataset, args.dataset, args.num_labeled, args.seed)

    def create_model(args):
        if args.dataset == 'mnist':
            import models.inferenceModel as models
            model = models.build_InferenceModel(args)
        elif args.dataset == 'cifar':
            import models.resnet as models
            model = models.ResNet50(args.num_classes)

        return model

    device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed is not None:
        set_seed(args)

    args.logs = defaultdict(OrderedDict)

    if args.dataset == 'cifar':
        args.num_classes = 100
        args.in_dim = (3, 32, 32)

    elif args.dataset == 'mnist':
        args.num_classes = 10
        args.in_dim = (1, 28, 28)
        args.model_cardinality = 4
        args.model_depth = 28
        args.model_width = 4

    if args.dataset == 'cifar':
        _, _, unlabeled_dataset, _, _ = get_cifar10(args)
        val_train_dataset, val_test_dataset, _, test_dataset, finetune_dataset = get_cifar100(args)
    elif args.dataset == 'mnist':
        val_train_dataset, val_test_dataset, unlabeled_dataset, test_dataset, finetune_dataset = get_mnist(args)

    train_sampler = RandomSampler

    val_train_loader = DataLoader(
        val_train_dataset,
        sampler=train_sampler(val_train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    finetune_loader = DataLoader(
        finetune_dataset,
        sampler=train_sampler(finetune_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    val_test_loader = DataLoader(
        val_test_dataset,
        sampler=train_sampler(val_test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(args).to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    from models.ema import ModelEMA
    ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0
    model.zero_grad()
    train(args, val_train_loader, val_test_loader, unlabeled_trainloader, test_loader, finetune_loader, model, optimizer, ema_model, scheduler)


def train(args, val_train_loader, val_test_loader, unlabeled_trainloader, test_loader, finetune_loader, model, optimizer, ema_model, scheduler):
    end = time.time()

    labeled_iter = iter(finetune_loader)
    unlabeled_iter = iter(unlabeled_trainloader)

    args.t=0
    model.pretrain()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(finetune_loader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            del logits
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            if args.sup_only: loss = Lx
            else: loss = Lx + args.lambda_u * Lu
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            if batch_idx ==0: print('%d: epoch: %d / %d. Time: %s' % (args.num_labeled, epoch+1, args.epochs, datetime.now()-args.beginning))
            args.t+=1

        test_model = ema_model.ema if args.use_ema else model

        test_loss, test_acc = test(args, test_loader, test_model)
        val_test_loss, val_test_acc = test(args, val_test_loader, test_model)
        val_train_loss, val_train_acc = test(args, val_train_loader, test_model)

        args.logs['train_loss'][args.t] = losses.avg
        args.logs['train_loss_sup'][args.t] = losses_x.avg
        args.logs['train_loss_unsup'][args.t] = losses_u.avg
        args.logs['val_train_loss'][args.t] = val_train_loss
        args.logs['val_train_acc'][args.t] = val_train_acc
        args.logs['val_test_loss'][args.t] = val_test_loss
        args.logs['val_test_acc'][args.t] = val_test_acc
        args.logs['test_loss'][args.t] = test_loss
        args.logs['test_acc'][args.t] = test_acc
        console_log(args, main_train=False)

        name = "./%s.pytorch" % (str(args.out))
        torch.save({"args.logs": args.logs}, name)


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy_fixmatch(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg


if __name__ == '__main__':
    for _, seed in enumerate([0,1,2,3,4]):
        main(seed)
