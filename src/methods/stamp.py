import math
import random
from copy import deepcopy

import torch.nn as nn
import torch
from src.utils.utils import softmax_entropy
import torch.nn.functional as F


def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)


class RBM:
    def __init__(self, max_len, num_class):
        self.num_class = num_class
        self.count_class = torch.zeros(num_class)
        self.data = [[] for _ in range(num_class)]
        self.max_len = max_len
        self.total_num = 0

    def remove_item(self):
        max_count = 0
        for i in range(self.num_class):
            if len(self.data[i]) == 0:
                continue
            if self.count_class[i] > max_count:
                max_count = self.count_class[i]
        max_classes = []
        for i in range(self.num_class):
            if self.count_class[i] == max_count and len(self.data[i]) > 0:
                max_classes.append(i)
        remove_class = random.choice(max_classes)
        self.data[remove_class].pop(0)

    def append(self, items, class_ids):
        for item, class_id in zip(items, class_ids):
            if self.total_num < self.max_len:
                self.data[class_id].append(item)
                self.total_num += 1
            else:
                self.remove_item()
                self.data[class_id].append(item)

    def get_data(self):
        data = []
        for cls in range(self.num_class):
            data.extend(self.data[cls])
            self.count_class[cls] = 0.9 * self.count_class[cls] + 0.1 * len(self.data[cls])
        return torch.stack(data)

    def __len__(self):
        return self.total_num

    def reset(self):
        self.count_class = torch.zeros(self.num_class)
        self.data = [[] for _ in range(self.num_class)]
        self.total_num = 0


class FIFO:
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = None

    def append(self, item):
        if self.data is None:
            self.data = item
        else:
            self.data = torch.cat([self.data, item], dim=0)
        if len(self.data) > self.max_len:
            self.data = self.data[-self.max_len:]

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def reset(self):
        self.data = None


class STAMP(nn.Module):
    def __init__(self, model, optimizer, alpha, num_class):
        super(STAMP, self).__init__()
        self.model = self.configure_model(model)
        self.norm_model = deepcopy(self.model).train()
        self.num_class = num_class
        self.optimizer = optimizer
        self.alpha = alpha
        self.margin = alpha * math.log(num_class)
        self.mem = RBM(64, num_class)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        if num_class == 1000:
            self.max_iter = 750
        else:
            self.max_iter = 150
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                # m.momentum = 0.2
                m.running_mean = None
                m.running_var = None
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'conv5_x' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def forward(self, x):
        output = self.update_memory(x)
        if len(self.mem) != 0:
            self.adapt()
        return output

    def update_memory(self, x):
        x_origin = x[0]
        if self.num_class == 1000:
            outputs = []
            output_origin = self.model(x_origin)
            outputs.append(output_origin.softmax(dim=1))

            for i in range(1, len(x)):
                x_aug = x[i]
                outputs.append(self.model(x_aug).softmax(dim=1))
            output = torch.stack(outputs, dim=0)
            output = torch.mean(output, dim=0)

            entropys = entropy(output)
            filter_ids = torch.where(entropys < self.margin)
            x_append = x_origin[filter_ids]
            self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids])
        else:
            outputs = []
            self.model.train()
            output_origin = self.model(x_origin)
            output_norm = self.norm_model(x_origin)
            filter_ids_0 = torch.where(output_origin.max(dim=1)[1] == output_norm.max(dim=1)[1])
            outputs.append(output_origin.softmax(dim=1))
            for i in range(1, len(x)):
                x_aug = x[i]
                outputs.append(self.model(x_aug).softmax(dim=1))
            output = torch.stack(outputs, dim=0)
            output = torch.mean(output, dim=0)
            entropys = entropy(output)[filter_ids_0]
            filter_ids = torch.where(entropys < self.margin)
            x_append = x_origin[filter_ids_0][filter_ids]
            self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids_0][filter_ids])
        return output, -entropy(output)

    @torch.enable_grad()
    def adapt(self):
        data = self.mem.get_data()
        self.optimizer.zero_grad()
        # data = x_origin
        if len(data) > 0:
            output_1 = self.model(data)
            entropys = softmax_entropy(output_1)
            # coeff = 1 / (torch.exp(entropys.clone().detach() - self.margin))
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second time forward
            output_1 = self.model(data)
            entropys = softmax_entropy(output_1)
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.mem.reset()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_iter)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
