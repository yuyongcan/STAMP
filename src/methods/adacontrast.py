"""
Builds upon: https://github.com/DianCh/AdaContrast
Corresponding paper: https://arxiv.org/abs/2204.10377
"""

import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from ..models import OfficeHome_Shot
from ..models.base_model import BaseModel


class AdaMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a memory bank
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
            self,
            src_model,
            momentum_model,
            K=16384,
            m=0.999,
            T_moco=0.07,
            checkpoint_path=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: buffer size; number of keys
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdaMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # create the fc heads
        feature_dim = src_model.output_dim

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K))
        self.register_buffer("mem_labels", torch.randint(0, src_model.num_classes, (K,)))
        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name[len("module."):] if name.startswith("module.") else name
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
                self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels):
        """
        Update features and corresponding pseudo labels
        """

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            feats_q: <B, D> query image features before normalization
            logits_q: <B, C> logits for class prediction from queries
            logits_ins: <B, K> logits for instance prediction
            k: <B, D> contrastive keys
        """

        # compute query features
        feats_q, logits_q = self.src_model(im_q, return_feats=True)

        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, _ = self.momentum_model(im_k, return_feats=True)
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k


class AdaContrast(nn.Module):
    def __init__(self, model, optimizer, steps, episodic, arch_name, queue_size, momentum, temperature, contrast_type,
                 ce_type, alpha, beta, eta,
                 dist_type, ce_sup_type, refine_method, num_neighbors):
        super().__init__()
        # TTA base init
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # Hyperparameters
        self.queue_size = queue_size
        self.m = momentum
        self.T_moco = temperature

        self.contrast_type = contrast_type
        self.ce_type = ce_type
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        self.dist_type = dist_type
        self.ce_sup_type = ce_sup_type
        self.refine_method = refine_method
        self.num_neighbors = num_neighbors

        self.first_X_samples = 0

        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.src_model = BaseModel(model, arch_name)

        # Setup EMA model
        self.momentum_model = self.copy_model(self.src_model)

        self.model = AdaMoCo(
            src_model=self.src_model,
            momentum_model=self.momentum_model,
            K=self.queue_size,
            m=self.m,
            T_moco=self.T_moco,
        ).cuda()

        self.banks = {
            "features": torch.tensor([], device="cuda"),
            "probs": torch.tensor([], device="cuda"),
            "ptr": 0
        }

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.momentum_model]
        self.model_states, self.optimizer_state = \
            self.copy_model_and_optimizer()

    def forward(self, x):
        images_test, images_w, images_q, images_k = x

        # Train model
        self.model.train()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        # Create the final output prediction
        self.model.eval()
        _, outputs = self.model(images_test, cls_only=True)
        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        :param x: The buffered data created with a sliding window
        :return: Dummy output. Has no effect
        """
        imgs_test = x[0]
        return torch.zeros_like(imgs_test)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        _, images_w, images_q, images_k = x

        self.model.train()
        feats_w, logits_w = self.model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            if self.first_X_samples >= 1024:
                self.refine_method = "nearest_neighbors"
            else:
                self.refine_method = None
                self.first_X_samples += len(feats_w)

            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, self.banks, self.refine_method, self.dist_type, self.num_neighbors
            )

        _, logits_q, logits_ins, keys = self.model(images_q, images_k)
        # update key features and corresponding pseudo labels
        self.model.update_memory(keys, pseudo_labels_w)

        # moco instance discrimination
        loss_ins, _ = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=self.model.mem_labels,
            contrast_type=self.contrast_type,
        )

        # classification
        loss_cls, _ = classification_loss(
            logits_w, logits_q, pseudo_labels_w, self.ce_sup_type
        )

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, self.ce_sup_type)
            if self.eta > 0
            else torch.tensor([0.0]).to("cuda")
        )

        loss = (
                self.alpha * loss_cls
                + self.beta * loss_ins
                + self.eta * loss_div
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = self.model.momentum_model(images_w, return_feats=True)

        self.update_labels(feats_w, logits_w)

        return logits_q

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters.

        Walk the model's modules and collect all parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def copy_model(model):
        if isinstance(model, BaseModel):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()

        self.model = AdaMoCo(
            src_model=self.src_model,
            momentum_model=self.momentum_model,
            K=self.queue_size,
            m=self.m,
            T_moco=self.T_moco,
        ).cuda()
        self.first_X_samples = 0
        self.banks = {
            "features": torch.tensor([], device="cuda"),
            "probs": torch.tensor([], device="cuda"),
            "ptr": 0
        }

    @torch.no_grad()
    def update_labels(self, features, logits):
        # 1) avoid inconsistency among DDP processes, and
        # 2) have better estimate with more data points

        probs = F.softmax(logits, dim=1)

        start = self.banks["ptr"]
        end = start + len(features)
        if self.banks["features"].shape[0] < self.queue_size:
            self.banks["features"] = torch.cat([self.banks["features"], features], dim=0)
            self.banks["probs"] = torch.cat([self.banks["probs"], probs], dim=0)
            self.banks["ptr"] = end % len(self.banks["features"])
        else:
            idxs_replace = torch.arange(start, end).cuda() % len(self.banks["features"])
            self.banks["features"][idxs_replace, :] = features
            self.banks["probs"][idxs_replace, :] = probs
            self.banks["ptr"] = end % len(self.banks["features"])

    @staticmethod
    def configure_model(model):
        """Configure model"""
        model.train()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
        return model


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, dist_type, num_neighbors):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


@torch.no_grad()
def refine_predictions(
        features,
        probs,
        banks,
        refine_method,
        dist_type,
        num_neighbors,
        gt_labels=None,
):
    if refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, dist_type, num_neighbors
        )
    elif refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy


def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = None

    return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, ce_sup_type):
    if ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels)
        accuracy = None
    elif ce_sup_type == "weak_strong":
        loss_cls = cross_entropy_loss(logits_s, target_labels)
        accuracy = None
    else:
        raise NotImplementedError(
            f"{ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def diversification_loss(logits_w, logits_s, ce_sup_type):
    if ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss


def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances
