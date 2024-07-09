import argparse
import os
import os.path as osp
import random

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


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.enable = True
    parser = argparse.ArgumentParser(description='train_source_os')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=200, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='cifar10',
                        choices=['VISDA-C', 'office', 'officehome', 'office-caltech', 'domainnet126', 'cifar10',
                                 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--net', type=str, default='ResNet50_10',
                        help="vgg16, ResNet50_10, resnet101, vit, WideResNet_8,ResNet18_8")
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
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['cosine', 'multistep'])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--steps', type=list, default=[60, 120, 160])
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='train',
                                               transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)
    # pass

    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker,
                                               split='train', transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)

    model_name = args.net
    ckpt_path = os.path.join(args.ckpt, 'models', model_name + '.pt')

    # train_idx = np.where(np.array(train_dataset.targets) < 8)[0]
    # test_idx = np.where(np.array(test_dataset.targets) < 8)[0]
    # train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    # test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.worker)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                           num_workers=args.worker)
    # if args.dset == 'cifar10':
    model = load_model(model_name).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    if args.scheduler == 'cosine':
        args.max_epoch = 200
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=args.max_epoch)
    elif args.scheduler == 'multistep':
        args.max_epoch = 200
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=args.steps,
                                                         gamma=args.gamma)
    CELOSS = nn.CrossEntropyLoss()
    model.train()
    max_acc = 0
    for epoch in range(args.max_epoch):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            loss = CELOSS(model(inputs), labels)

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('epoch:%d,iter:%d,loss:%.4f' % (epoch, i, loss.item()))
        model.eval()
        acc = 0

        scheduler.step()
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
