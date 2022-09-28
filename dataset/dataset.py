import logging
import math

import numpy as np
import torch.utils.data
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset

from .randaugment import RandAugmentMNIST_fixmatch
from .randaugment import RandAugmentCIFAR_fixmatch
from .augmentation import RandAugmentMNIST_mpl
from .augmentation import RandAugmentCIFAR_mpl
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def get_mnist(args):
    if args.baseline == 'fixmatch': transform_unlabeled = TransformFixMatch_mnist()
    elif args.baseline == 'pham': transform_unlabeled = TransformMPL_mnist(args)
    elif args.baseline == 'mt': transform_unlabeled = transforms.ToTensor()

    base_dataset = datasets.MNIST('./root', train=True, download=True)
    base_dataset.data = base_dataset.data.numpy()
    base_dataset.targets = base_dataset.targets.tolist()

    val_train_idxs, val_test_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets, 10)

    train_unlabeled_dataset = MNISTSSL('./root', train_unlabeled_idxs, train=True, transform=transform_unlabeled)
    val_train_dataset = MNISTSSL('./root', val_train_idxs, train=True, transform=transforms.ToTensor())
    finetune_dataset = MNISTSSL('./root', val_train_idxs, train=True, transform=transforms.ToTensor())
    val_test_dataset = MNISTSSL('./root', val_test_idxs, train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./root', train=False, transform=transforms.ToTensor(), download=False)

    return val_train_dataset, val_test_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    if args.baseline == 'fixmatch': transform_unlabeled = TransformFixMatch_cifar(mean=cifar10_mean, std=cifar10_std)
    elif args.baseline == 'pham': transform_unlabeled = TransformMPL_cifar(args, mean=cifar10_mean, std=cifar10_std)
    elif args.baseline == 'mt': transform_unlabeled = transform_val

    base_dataset = datasets.CIFAR10('./root', train=True, download=True)

    val_train_idxs, val_test_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets, 10)

    train_unlabeled_dataset = CIFAR10SSL('./root', train_unlabeled_idxs, train=True, transform=transform_unlabeled)
    val_train_dataset = CIFAR10SSL('./root', val_train_idxs, train=True, transform=transform_val)
    finetune_dataset = CIFAR10SSL('./root', val_train_idxs, train=True, transform=transform_labeled)
    val_test_dataset = CIFAR10SSL('./root', val_test_idxs, train=True, transform=transform_val)
    test_dataset = datasets.CIFAR10('./root', train=False, transform=transform_val, download=False)

    return val_train_dataset, val_test_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_cifar100(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    if args.baseline == 'fixmatch': transform_unlabeled = TransformFixMatch_cifar(mean=cifar100_mean, std=cifar100_std)
    elif args.baseline == 'pham': transform_unlabeled = TransformMPL_cifar(args, mean=cifar100_mean, std=cifar100_std)
    elif args.baseline == 'mt': transform_unlabeled = transform_val

    base_dataset = datasets.CIFAR100('./root', train=True, download=True)

    val_train_idxs, val_test_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets, 100)

    val_train_dataset = CIFAR100SSL('./root', val_train_idxs, train=True, transform=transform_val)
    finetune_dataset = CIFAR100SSL('./root', val_train_idxs, train=True, transform=transform_labeled)
    val_test_dataset = CIFAR100SSL('./root', val_test_idxs, train=True, transform=transform_val)
    train_unlabeled_dataset = CIFAR100SSL('./root', train_unlabeled_idxs, train=True, transform=transform_unlabeled)
    test_dataset = datasets.CIFAR100('./root', train=False, transform=transform_val, download=False)

    return val_train_dataset, val_test_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def x_u_split(args, labels, num_classes):
    label_per_class = args.num_labeled // num_classes
    labels = np.array(labels)
    val_train_idx = []
    val_test_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        val_train_idx.extend(idx[:int(0.85*len(idx))])
        val_test_idx.extend(idx[int(0.85*len(idx)):])
    val_train_idx = np.array(val_train_idx)
    val_test_idx = np.array(val_test_idx)

    if 0.85*args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        val_train_idx = np.hstack([val_train_idx for _ in range(num_expand_x)])
        val_test_idx = np.hstack([val_test_idx for _ in range(num_expand_x)])
    np.random.shuffle(val_train_idx)
    np.random.shuffle(val_test_idx)

    return val_train_idx, val_test_idx, unlabeled_idx


class TransformFixMatch_mnist(object):
    def __init__(self):
        self.weak = transforms.Compose([transforms.RandomCrop(size=28, padding=int(28*0.125), padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomCrop(size=28, padding=int(28*0.125), padding_mode='reflect'),
            RandAugmentMNIST_fixmatch(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatch_cifar(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformMPL_mnist(object):
    def __init__(self, args):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomCrop(size=28, padding=int(28 * 0.125), fill=128, padding_mode='constant')
        ])
        self.aug = transforms.Compose([
            transforms.RandomCrop(size=28, padding=int(28 * 0.125), fill=128, padding_mode='constant'),
            RandAugmentMNIST_mpl(n=n, m=m)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)

class TransformMPL_cifar(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), fill=128, padding_mode='constant')
        ])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), fill=128, padding_mode='constant'),
            RandAugmentCIFAR_mpl(n=n, m=m)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class MNISTSSL(datasets.MNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.data = self.data.numpy()
        self.targets = self.targets.tolist()

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            # print(type(self.data))
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
