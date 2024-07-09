import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.utils import compute_os_variance

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
            print('gtting source feature from ckpt')
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


def append_prototypes(pool, feat_ext, logit, ts, ts_pro):
    """
    Append strong OOD prototypes to the prototype pool.

    Parameters:
        pool : Prototype pool.
        feat_ext : Normalized features of the input images.
        logit : Cosine similarity between the features and the weak OOD prototypes.
        ts : Threshold to separate weak and strong OOD samples.
        ts_pro : Threshold to append strong OOD prototypes.

    """
    added_list = []
    update = 1

    while update:
        feat_mat = pool(F.normalize(feat_ext), all=True)
        if not feat_mat == None:
            new_logit = torch.cat([logit, feat_mat], 1)
        else:
            new_logit = logit

        r_i_pro, _ = new_logit.max(dim=-1)

        r_i, _ = logit.max(dim=-1)

        if added_list != []:
            for add in added_list:
                # if added_list is not empty, set the cosine similarity between the added features and the strong OOD prototypes to 1, to avoid the added features to be appended to the prototype pool again.
                r_i[add] = 1
        min_logit, min_index = r_i.min(dim=0)

        if (1 - min_logit) > ts:
            # if the cosine similarity between the feature and the weak OOD prototypes is less than the threshold ts, the feature is a strong OOD sample.
            added_list.append(min_index)
            if (1 - r_i_pro[min_index]) > ts_pro:
                # if this strong OOD sample is far away from all the strong OOD prototypes, append it to the prototype pool.
                pool.update_pool(F.normalize(feat_ext[min_index].unsqueeze(0)))
        else:
            # all the features are weak OOD samples, stop the loop.
            update = 0


class OWTTT(nn.Module):
    def __init__(self, model, optimizer, class_num=10, ce_scale=0.2, da_scale=1, delta=0.1, queue_length=512,
                 max_prototypes=100,
                 source_memory=None, source_distribution=None):
        super(OWTTT, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.model_state, self.optimizer_state = \
            self.copy_model_and_optimizer()
        self.ood_memory = Prototype_Pool(max=max_prototypes)
        self.queue_training = []
        self.queue_inference = []
        self.ema_total_n = 0.
        self.ema_distribution = {}
        self.ema_distribution['mu'] = torch.zeros(self.model.output_dim).float()
        self.ema_distribution['cov'] = torch.zeros(self.model.output_dim, self.model.output_dim).float()

        self.ce_scale = ce_scale
        self.da_scale = da_scale
        self.delta = delta
        self.queue_length = queue_length
        self.max_prototypes = max_prototypes
        self.source_memory = source_memory  # source prototypes
        bias = source_distribution['cov'].max().item() / 30.
        self.template_cov = torch.eye(self.model.output_dim).cuda() * bias
        self.source_distribution = source_distribution  # a dict containing the mu and cov matrix of the source domain

        self.class_num = class_num
        if class_num == 10:
            self.loss_scale = 0.05
        else:
            self.loss_scale = 0.05

    @torch.enable_grad()
    def forward(self, inputs):
        return self.adpat_and_forward(inputs)

    def adpat_and_forward(self, inputs):
        self.model.eval()
        threshold_range = np.arange(0, 1, 0.01)  # use for consequent operation

        feat, _ = self.model(inputs, return_feats=True)
        feat_norm = F.normalize(feat)
        self.optimizer.zero_grad()

        cos_sim_src = self.source_memory(feat_norm, all=True)

        cos_sim_ood = self.ood_memory(feat_norm)

        if cos_sim_ood is not None:
            cos_sim = torch.cat([cos_sim_src, cos_sim_ood], dim=1)
        else:
            cos_sim = cos_sim_src

        logits = cos_sim / self.delta

        ood_score, pseudo_labels = 1 - cos_sim.max(dim=-1)[0], cos_sim.max(dim=-1)[1]
        ood_score_src = 1 - cos_sim_src.max(dim=-1)[0]

        self.queue_training.extend(ood_score_src.detach().cpu().tolist())
        self.queue_training = self.queue_training[-self.queue_length:]

        criterias = [compute_os_variance(np.array(self.queue_training), th) for th in threshold_range]
        best_threshold_ood = threshold_range[np.argmin(criterias)]
        seen_mask = (ood_score_src < best_threshold_ood)
        unseen_mask = (ood_score_src >= best_threshold_ood)

        if unseen_mask.sum().item() != 0:
            criterias = [compute_os_variance(ood_score[unseen_mask].detach().cpu().numpy(), th) for th in
                         threshold_range]
            best_threshold_exp = threshold_range[np.argmin(criterias)]

            # append new strong OOD prototypes to the prototype pool.
            append_prototypes(self.ood_memory, feat, cos_sim_src, best_threshold_ood, best_threshold_exp)

        len_memory = len(cos_sim[0])
        if len_memory != self.class_num:
            if seen_mask.sum().item() != 0:
                pseudo_labels[seen_mask] = cos_sim_src[seen_mask].max(dim=-1)[1]
            if unseen_mask.sum().item() != 0:
                pseudo_labels[unseen_mask] = self.class_num
        else:
            pseudo_labels = cos_sim_src[seen_mask].max(dim=-1)[1]

        loss = torch.tensor(0.).cuda()
        # ------distribution alignment------
        if seen_mask.sum().item() != 0:
            self.model.train()
            feat_global = self.model(inputs, return_feats=True)[0]
            # Global Gaussian
            b = feat_global.shape[0]
            self.ema_total_n += b
            alpha = 1. / 1280 if self.ema_total_n > 1280 else 1. / self.ema_total_n
            delta_pre = (feat_global - self.ema_distribution['mu'].cuda())
            delta = alpha * delta_pre.sum(dim=0)
            tmp_mu = self.ema_distribution['mu'].cuda() + delta
            tmp_cov = self.ema_distribution['cov'].cuda() + alpha * (
                    delta_pre.t() @ delta_pre - b * self.ema_distribution['cov'].cuda()) - delta[:, None] @ delta[None,
                                                                                                            :]
            with torch.no_grad():
                self.ema_distribution['mu'] = tmp_mu.detach().cpu()
                self.ema_distribution['cov'] = tmp_cov.detach().cpu()

            source_domain = torch.distributions.MultivariateNormal(self.source_distribution['mu'],
                                                                   self.source_distribution['cov'] + self.template_cov)
            target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + self.template_cov)
            loss += self.da_scale * (
                    torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(
                target_domain, source_domain)) * self.loss_scale

        if len_memory != self.class_num and seen_mask.sum().item() != 0 and unseen_mask.sum().item() != 0:
            a, idx1 = torch.sort((ood_score_src[seen_mask]), descending=True)
            filter_down = a[-int(seen_mask.sum().item() * (1 / 2))]
            a, idx1 = torch.sort((ood_score_src[unseen_mask]), descending=True)
            filter_up = a[int(unseen_mask.sum().item() * (1 / 2))]
            for j in range(len(pseudo_labels)):
                if ood_score_src[j] >= filter_down and seen_mask[j]:
                    seen_mask[j] = False
                if ood_score_src[j] <= filter_up and unseen_mask[j]:
                    unseen_mask[j] = False

        if len_memory != self.class_num:
            entropy_seen = nn.CrossEntropyLoss()(logits[seen_mask, :self.class_num], pseudo_labels[seen_mask])
            entropy_unseen = nn.CrossEntropyLoss()(logits[unseen_mask], pseudo_labels[unseen_mask])
            loss += self.ce_scale * (entropy_seen + entropy_unseen) / 2 

        try:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        except:
            print('can not backward')
        torch.cuda.empty_cache()

        ####-------------------------- Test ----------------------------####
        with torch.no_grad():
            self.model.eval()
            feats = self.model(inputs, return_feats=True)[0]
            cos_sim_src = self.source_memory(F.normalize(feats), all=True)
            softmax_logit = (cos_sim_src / self.delta).softmax(dim=-1)
            # predicted = softmax_logit.max(dim=-1)[1]
            # ood_score_src = 1 - cos_sim_src.max(dim=-1)[0]
            #
            # self.queue_inference.extend(ood_score_src.detach().cpu().tolist())
            # self.queue_inference = self.queue_inference[-self.queue_length:]
            #
            # criterias = [compute_os_variance(np.array(self.queue_inference), th) for th in threshold_range]
            # best_threshold_ood = threshold_range[np.argmin(criterias)]
            # seen_mask = (ood_score_src < best_threshold_ood)
            # unseen_mask = (ood_score_src >= best_threshold_ood)
            # predicted[unseen_mask] = self.class_num
            #
            # one = torch.ones(inputs.shape[0]) * self.class_num
            # predicted = torch.where(predicted > self.class_num - 1, one.cuda(), predicted)
        return softmax_logit, cos_sim_src.max(dim=-1)[0]

    def copy_model_and_optimizer(self):
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.ood_memory = Prototype_Pool(max=self.max_prototypes)
        self.queue_training = []
        self.queue_inference = []
        self.ema_total_n = 0.
        self.ema_distribution = {}
        self.ema_distribution['mu'] = torch.zeros(self.model.output_dim).float()
        self.ema_distribution['cov'] = torch.zeros(self.model.output_dim).float()
