import torch
import torch.nn as nn
from ..utils import memory
from copy import deepcopy
from .bn import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule
from src.data.augmentations import get_tta_transforms

class RoTTA(nn.Module):
    def __init__(self, model, optimizer, dataset_name, MEMORY_SIZE, NUM_CLASS, LAMBDA_T, LAMBDA_U, NU, UPDATE_FREQUENCY,
                 ALPHA, steps=1):
        super(RoTTA, self).__init__()
        self.ALPHA = ALPHA
        self.optimizer = optimizer
        self.model = model
        self.MEMORY_SIZE = MEMORY_SIZE
        self.NUM_CLASS = NUM_CLASS
        self.LAMBDA_T = LAMBDA_T
        self.LAMBDA_U = LAMBDA_U
        self.steps = steps
        self.mem = memory.CSTU(capacity=MEMORY_SIZE, num_class=NUM_CLASS, lambda_t=LAMBDA_T, lambda_u=LAMBDA_U)
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(dataset_name)
        self.nu = NU
        self.update_frequency = UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    @staticmethod
    def collect_params(model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    @staticmethod
    def configure_model(model: nn.Module, ALPHA):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

        self.mem = memory.CSTU(capacity=self.MEMORY_SIZE, num_class=self.NUM_CLASS, lambda_t=self.LAMBDA_T,
                               lambda_u=self.LAMBDA_U)
        self.model_ema = self.build_ema(self.model)

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
