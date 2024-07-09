import argparse
import os
import os.path as osp
import random
import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from src.data.data import load_dataset
from src.models import load_model
from src.utils import loss
from src.utils.loss import CrossEntropyLabelSmooth

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

transform_test = transforms.Compose(
    [transforms.ToTensor()])

if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.enable = True
    parser = argparse.ArgumentParser(description='train_source_os')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=200, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='cifar100',
                        choices=['VISDA-C', 'office', 'officehome', 'office-caltech', 'domainnet126', 'cifar10',
                                 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--net', type=str, default='ResNet18_100',
                        help="vgg16, ResNet50_100, resnet101, ResNet18_100, Hendrycks2020AugMix_ResNeXt_80, ResNet18_80")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='../ckpt/models')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--data_dir', type=str, default='/data2/yongcan.yu/datasets')
    parser.add_argument('--ckpt', type=str, default='../ckpt')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--steps', type=list, default=[60, 120, 160, 200, 240])
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # pass

    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker,
                                               split='train', transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)

    model_name = args.net
    if model_name.split('_')[-1] == '80':
        train_idx = np.where(np.array(train_dataset.targets) < 80)[0]
        test_idx = np.where(np.array(test_dataset.targets) < 80)[0]
        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.worker)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.worker)
    ckpt_path = os.path.join(args.ckpt, 'models', model_name + '_1.pt')

    # if args.dset == 'cifar10':
    # model = load_model(model_name).cuda()
    model = load_model(model_name).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.scheduler == 'multistep':
        args.max_epoch = 250
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
    if args.scheduler == 'cosine':
        args.max_epoch = 200
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0)
    model.train()
    max_acc = 0
    for epoch in range(args.max_epoch):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            loss = F.cross_entropy(model(inputs), labels)

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('epoch:%d,iter:%d,loss:%.4f' % (epoch, i, loss.item()))
        train_scheduler.step()
        model.eval()
        acc = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            acc += (torch.max(outputs, 1)[1] == labels).float().sum().item()
        acc = acc / len(test_loader.dataset)
        print('epoch:%d,acc:%.4f' % (epoch, acc))
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        model.train()
    print('max_acc:%.4f' % max_acc)
