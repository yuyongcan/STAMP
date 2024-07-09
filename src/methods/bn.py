# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

"""Batch norm variants
AlphaBatchNorm builds upon: https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class AlphaBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = AlphaBatchNorm(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBatchNorm.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        assert alpha >= 0 and alpha <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = alpha

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)

    def forward(self, input):
        self.norm(input)

        running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
        running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


class EMABatchNorm(nn.Module):
    @staticmethod
    def adapt_model(model):
        model = EMABatchNorm(model)
        return model

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # store statistics, but discard result
        self.model.train()
        self.model(x)
        # store statistics, use the stored stats
        self.model.eval()
        return self.model(x)

class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias
