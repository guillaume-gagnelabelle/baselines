import torchvision.utils
from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import TensorDataset, Subset
import matplotlib.pyplot as plt
import numpy as np
import math


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()


def get_data(args):
    if args.dataset == "mnist":
        mnist_transforms1 = transforms.Compose([
            transforms.RandomResizedCrop(size=(28, 28)),
            transforms.ToTensor()
        ])

        mnist_transforms2 = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor()
        ])

        train_data_orig = datasets.MNIST("./root", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
        aug_data1 = datasets.MNIST("./root", train=True, transform=mnist_transforms1, target_transform=None, download=True)
        aug_data2 = datasets.MNIST("./root", train=True, transform=mnist_transforms2, target_transform=None, download=True)
        test_data = datasets.MNIST("./root", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
        val_data = train_data_orig

        args.num_classes = 10
        args.in_dim = (1, 28, 28)

    elif args.dataset == "cifar":
        cifar_transforms1 = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        cifar_transforms2 = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        raw_transforms_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
        raw_transforms_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

        train_data_orig = datasets.CIFAR10("./root", train=True, transform=raw_transforms_cifar10, download=True)
        aug_data1 = datasets.CIFAR10("./root", train=True, transform=cifar_transforms1, download=True)
        aug_data2 = datasets.CIFAR10("./root", train=True, transform=cifar_transforms2, download=True)
        test_data = datasets.CIFAR100("./root", train=False, transform=raw_transforms_cifar100, download=True)
        val_data = datasets.CIFAR100("./root", train=True, transform=raw_transforms_cifar100, download=True)

        args.num_classes = 100
        args.in_dim = (3, 32, 32)

    train_data = []
    for i in range(len(train_data_orig)):
        train_data.append([train_data_orig[i], aug_data1[i], aug_data2[i]])

    val_train_idxs, val_test_idxs, _ = x_u_split(args, val_data.targets, args.num_classes)
    val_train_data, val_test_data = Subset(val_data, val_train_idxs), Subset(val_data, val_test_idxs)

    return train_data, val_train_data, val_test_data, test_data


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