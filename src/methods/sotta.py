import random
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from ..utils.sam import SAM
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class HUS:
    def __init__(self, capacity, num_class, threshold=None):
        self.num_class = num_class
        self.data = [[[], [], []] for _ in
                     range(self.num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * self.num_class
        self.capacity = capacity
        self.threshold = threshold

    def get_memory(self):
        data = []
        for x in self.data:
            data.extend(x[0])
        data = torch.stack(data)
        return data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[2] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)
        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][1])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][1])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][3].append(aux[i])


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b

loss_fn = HLoss()

class SoTTA(nn.Module):
    def __init__(self, model, optimizer, memory_size, ConfThreshold, num_class):
        super(SoTTA, self).__init__()
        self.model = model
        self.optimizer = optimizer
        assert (isinstance(self.optimizer, SAM))
        self.memory_size = memory_size
        self.ConfThreshold = ConfThreshold
        self.num_class = num_class
        self.memory = HUS(memory_size, num_class, ConfThreshold)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

    def copy_model_and_optimizer(self, model, optimizer):
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)
        self.memory = HUS(self.memory_size, self.num_class, self.ConfThreshold)

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
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

    @staticmethod
    def configure_model(model, momentum=0.2):
        for param in model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                module.track_running_stats = True
                module.momentum = momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
        return model

    def forward(self, x):
        self.model.eval()
        logits = self.model(x)
        logits = logits.softmax(dim=1)
        confidences, pseudo_labels = torch.max(logits, dim=1)
        for i in range(x.shape[0]):
            self.memory.add_instance([x[i], pseudo_labels[i], confidences[i]])

        feats = self.memory.get_memory()
        try:
            if feats.shape[0] == 0:
                return logits
            elif feats.shape[0] == 1:
                self.model.eval()
            else:
                self.model.train()
        except:
            logger.warning("feats.shape[0] == 0")
            return logits
        self.optimize(feats)
        return logits

    @torch.enable_grad()
    def optimize(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = loss_fn(logits)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        self.model.train()
        logits = self.model(data)
        loss = loss_fn(logits)
        loss.backward()
        self.optimizer.second_step(zero_grad=True)

