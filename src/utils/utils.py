import argparse
import logging
import os
import random
import sys
from copy import deepcopy
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.weight_norm import WeightNorm
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items) / len(items)


def get_source_features_labels(model, source_loader, ckpt_dir=None):
    """
    Get the features of the source dataset.

    Parameters:
        model : The model used to extract features.
        source_loader : The source dataset loader.

    Returns:
        torch.Tensor: The features of the source dataset.
    """
    if ckpt_dir is not None:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        features_path = os.path.join(ckpt_dir, 'source_features.pt')
        labels_path = os.path.join(ckpt_dir, 'source_labels.pt')
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features = torch.load(features_path)
            labels = torch.load(labels_path)
            print('getting source feature from ckpt')
            return features, labels
    print('getting source feature')
    # model.eval()
    model.eval()
    features_all = []
    labels_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(source_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            features, logits = model(inputs, return_feats=True)
            # pseudo_label = logits.max(dim=1)[1]
            # acc = torch.sum(torch.argmax(logits, axis=1) == targets) / len(targets)
            # print('acc:', acc.item())

            features_all.append(features)
            # if not torch.all(mask):
            #     print('Warning: some pseudo labels are wrong!')
            #     raise Exception('Wrong pseudo labels!')
            labels_all.append(targets)
    features_all = torch.cat(features_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    print('source feature shape:', features_all.shape)
    if ckpt_dir is not None:
        torch.save(features_all, features_path)
        torch.save(labels_all, labels_path)
    return features_all, labels_all


class Prototype_Pool(nn.Module):
    """
    Prototype pool containing strong OOD prototypes.

    Methods:
        __init__: Constructor method to initialize the prototype pool, storing the values of delta, the number of weak OOD categories, and the maximum count of strong OOD prototypes.
        forward: Method to farward pass, return the cosine similarity with strong OOD prototypes.
        update_pool: Method to append and delete strong OOD prototypes.
    """

    def __init__(self, max=100, memory=None):
        super(Prototype_Pool, self).__init__()

        self.max_length = max
        self.flag = 0
        if memory is not None:
            self.register_buffer('memory', memory)
            self.flag = 1

    def forward(self, x, all=False):
        # if the flag is 0, the prototype pool is empty, return None.
        if not self.flag:
            return None

        # compute the cosine similarity between the features and the strong OOD prototypes.
        out = torch.mm(x, self.memory.t())

        if all == True:
            # if all is True, return the cosine similarity with all the strong OOD prototypes.
            return out
        else:
            # if all is False, return the cosine similarity with the nearest strong OOD prototype.
            return torch.max(out, dim=1)[0].unsqueeze(1)

    def update_pool(self, feature):
        if not self.flag:
            # if the flag is 0, the prototype pool is empty, use the feature to init the prototype pool.
            self.register_buffer('memory', feature.detach())
            self.flag = 1
        else:
            if self.memory.shape[0] < self.max_length:
                # if the number of strong OOD prototypes is less than the maximum count of strong OOD prototypes, append the feature to the prototype pool.
                self.memory = torch.cat([self.memory, feature.detach()], dim=0)
            else:
                # else then delete the earlest appended strong OOD prototype and append the feature to the prototype pool.
                self.memory = torch.cat([self.memory[1:], feature.detach()], dim=0)
        self.memory = F.normalize(self.memory)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)

    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device = None, cfg=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_all = []
    labels_all = []
    ood_scores_all = []
    num_class = cfg.num_classes
    samples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            samples += len(data[0])
            imgs, labels = data[0], data[1].cuda()
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            if isinstance(output, tuple):
                output, ood_score = output
            else:
                entropy = softmax_entropy(output)
                ood_score = -entropy
            outputs_all.append(output.cpu())
            labels_all.append(labels.cpu())
            ood_scores_all.append(ood_score.cpu())
        outputs_all = torch.cat(outputs_all, dim=0)
        pred_all = torch.argmax(outputs_all, dim=1)
        labels_all = torch.cat(labels_all, dim=0)
        ood_scores_all = torch.cat(ood_scores_all, dim=0)
        id_label = (labels_all <= num_class - 1) * (labels_all >= 0)
        ID_pred = pred_all[id_label]
        ID_label = labels_all[id_label]
        acc = torch.mean((ID_pred == ID_label).float()).item()
        auc = roc_auc_score(id_label.cpu().numpy(), ood_scores_all.cpu().numpy())
    return acc, auc


def split_up_model(model):
    modules = list(model.children())[:-1]
    classifier = list(model.children())[-1]
    while not isinstance(classifier, nn.Linear):
        sub_modules = list(classifier.children())[:-1]
        modules.extend(sub_modules)
        classifier = list(classifier.children())[-1]
    featurizer = nn.Sequential(*modules)

    return featurizer, classifier


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_output(encoder, classifier, x, arch):
    x = encoder(x)
    if arch == 'WideResNet':
        x = F.avg_pool2d(x, 8)
        features = x.view(-1, classifier.in_features)
    elif arch == 'vit':
        features = x[:, 0]
    else:
        features = x.squeeze()
    return features, classifier(features)


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cal_acc(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0])

    return accuracy * 100


def del_wn_hook(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)


def restore_wn_hook(model, name='weight'):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, name)


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


def deepcopy_model(model):
    ### del weight norm hook
    del_wn_hook(model)

    ### copy model
    model_cp = deepcopy(model)

    ### restore weight norm hook
    restore_wn_hook(model)
    restore_wn_hook(model_cp)

    return model_cp


def compute_os_variance(os, th):
    """
    Calculate the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """

    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--MODEL_CONTINUAL', default=None, type=str)
    parser.add_argument('--OPTIM_LR', default=None, type=float)
    parser.add_argument('--BN_ALPHA', default=None, type=float)
    parser.add_argument('--output_dir', default=None, type=str, help='path to output_dir file')
    parser.add_argument('--COTTA_RST', default=None, type=float)
    parser.add_argument('--COTTA_AP', default=None, type=float)
    parser.add_argument('--M_TEACHER_MOMENTUM', default=None, type=float)
    parser.add_argument('--EATA_DM', default=None, type=float)
    parser.add_argument('--EATA_FISHER_ALPHA', default=None, type=float)
    parser.add_argument('--EATA_E_MARGIN_COE', default=None, type=float)
    parser.add_argument('--T3A_FILTER_K', default=None, type=int)
    parser.add_argument('--LAME_AFFINITY', default=None, type=str)
    parser.add_argument('--LAME_KNN', default=None, type=int)

    parser.add_argument('--TEST_EPOCH', default=None, type=int)
    parser.add_argument('--SHOT_CLS_PAR', default=None, type=float)
    parser.add_argument('--SHOT_ENT_PAR', default=None, type=float)
    parser.add_argument('--NRC_K', default=None, type=int)
    parser.add_argument('--NRC_KK', default=None, type=int)
    parser.add_argument('--SAR_RESET_CONSTANT', default=None, type=float)
    parser.add_argument('--SAR_E_MARGIN_COE', default=None, type=float)
    parser.add_argument('--PLUE_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_QUEUE_SIZE', default=None, type=int)
    parser.add_argument('--TEST_BATCH_SIZE', default=None, type=int)
    parser.add_argument('--prediction_dir', default='./predictions', type=str, help='path to prediction_dir file')
    parser.add_argument('--CORRUPTION_NUM_OOD_SAMPLES', default=None, type=int)
    parser.add_argument('--OWTTT_CE_SCALE', default=None, type=float)
    parser.add_argument('--OWTTT_DA_SCALE', default=None, type=float)
    parser.add_argument('--SOTTA_THRESHOLD', default=None, type=float)
    parser.add_argument('--ODIN_TEMP', default=None, type=float)
    parser.add_argument('--ODIN_MAG', default=None, type=float)
    parser.add_argument('--STAMP_ALPHA', default=None, type=float)
    args = parser.parse_args()
    return args


def merge_cfg_from_args(cfg, args):
    if args.MODEL_CONTINUAL is not None:
        cfg.MODEL.CONTINUAL = args.MODEL_CONTINUAL
    if args.OPTIM_LR is not None:
        cfg.OPTIM.LR = args.OPTIM_LR
    if args.BN_ALPHA is not None:
        cfg.BN.ALPHA = args.BN_ALPHA
    if args.COTTA_RST is not None:
        cfg.COTTA.RST = args.COTTA_RST
    if args.COTTA_AP is not None:
        cfg.COTTA.AP = args.COTTA_AP
    if args.M_TEACHER_MOMENTUM is not None:
        cfg.M_TEACHER.MOMENTUM = args.M_TEACHER_MOMENTUM
    if args.EATA_DM is not None:
        cfg.EATA.D_MARGIN = args.EATA_DM
    if args.EATA_FISHER_ALPHA is not None:
        cfg.EATA.FISHER_ALPHA = args.EATA_FISHER_ALPHA
    if args.EATA_E_MARGIN_COE is not None:
        cfg.EATA.E_MARGIN_COE = args.EATA_E_MARGIN_COE
    if args.T3A_FILTER_K is not None:
        cfg.T3A.FILTER_K = args.T3A_FILTER_K
    if args.LAME_AFFINITY is not None:
        cfg.LAME.AFFINITY = args.LAME_AFFINITY
    if args.LAME_KNN is not None:
        cfg.LAME.KNN = args.LAME_KNN
    if args.TEST_EPOCH is not None:
        cfg.TEST.EPOCH = args.TEST_EPOCH
    if args.SHOT_CLS_PAR is not None:
        cfg.SHOT.CLS_PAR = args.SHOT_CLS_PAR
    if args.SHOT_ENT_PAR is not None:
        cfg.SHOT.ENT_PAR = args.SHOT_ENT_PAR
    if args.NRC_K is not None:
        cfg.NRC.K = args.NRC_K
    if args.NRC_KK is not None:
        cfg.NRC.KK = args.NRC_KK
    if args.SAR_RESET_CONSTANT is not None:
        cfg.SAR.RESET_CONSTANT = args.SAR_RESET_CONSTANT
    if args.SAR_E_MARGIN_COE is not None:
        cfg.SAR.E_MARGIN_COE = args.SAR_E_MARGIN_COE
    if args.PLUE_NUM_NEIGHBORS is not None:
        cfg.PLUE.NUM_NEIGHBORS = args.PLUE_NUM_NEIGHBORS
    if args.ADACONTRAST_NUM_NEIGHBORS is not None:
        cfg.ADACONTRAST.NUM_NEIGHBORS = args.ADACONTRAST_NUM_NEIGHBORS
    if args.ADACONTRAST_QUEUE_SIZE is not None:
        cfg.ADACONTRAST.QUEUE_SIZE = args.ADACONTRAST_QUEUE_SIZE
    if args.TEST_BATCH_SIZE is not None:
        cfg.TEST.BATCH_SIZE = args.TEST_BATCH_SIZE
    if args.CORRUPTION_NUM_OOD_SAMPLES is not None:
        cfg.CORRUPTION.NUM_OOD_SAMPLES = args.CORRUPTION_NUM_OOD_SAMPLES
    if args.OWTTT_DA_SCALE is not None:
        cfg.OWTTT.DA_SCALE = args.OWTTT_DA_SCALE
    if args.OWTTT_CE_SCALE is not None:
        cfg.OWTTT.CE_SCALE = args.OWTTT_CE_SCALE
    if args.SOTTA_THRESHOLD is not None:
        cfg.SOTTA.THRESHOLD = args.SOTTA_THRESHOLD
    if args.ODIN_TEMP is not None:
        cfg.ODIN.TEMP = args.ODIN_TEMP
    if args.ODIN_MAG is not None:
        cfg.ODIN.MAG = args.ODIN_MAG
    if args.STAMP_ALPHA is not None:
        cfg.STAMP.ALPHA = args.STAMP_ALPHA
