import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


class FeatMem(nn.Module):
    def __init__(self, memory, max_length, alpha):
        super(FeatMem, self).__init__()
        self.mem = memory
        self.class_num = len(memory)
        self.feature_dim = memory[0].shape[1]
        self.max_length = max_length
        # self.alphas = torch.ones((self.class_num, self.feature_num)) * alpha

    def forward(self, features, pseudo_labels, entropy):
        mem = F.normalize(torch.stack([self.mem[i].mean(dim=0) for i in range(self.class_num)]))
        cos = torch.matmul(features, mem.t())
        max_cos, max_idx = torch.max(cos, dim=1)
        filter = (max_cos > max_cos.median()) * (max_idx == pseudo_labels) * (entropy < entropy.median())
        # print('filter: ', filter.sum().item(), '/', len(filter))
        # self.update_mem(features[filter], pseudo_labels[filter])
        return max_cos,max_cos[filter]

    def update_mem(self, features, pseudo_labels):
        for i in range(len(pseudo_labels)):
            pseudo_label = pseudo_labels[i]
            feature = features[i].unsqueeze(0)
            self.mem[pseudo_label] = torch.cat([self.mem[pseudo_label], feature], dim=0)[-self.max_length:]
        return

    def update_cos(self):
        for i in range(self.class_num):
            mem_center = F.normalize(torch.unsqueeze(self.mem[i].mean(dim=0), 0))
            cos = torch.matmul(self.mem[i], mem_center.t()).squeeze()
            self.alphas[i] = cos
        return


class MEM(nn.Module):
    def __init__(self, model, memory):
        super(MEM, self).__init__()
        self.model = model
        self.memory = memory
        self.memory_cp = deepcopy(memory)
        self.model_cp = deepcopy(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

    @torch.enable_grad()
    def forward(self, inputs):
        # with torch.no_grad():
        feats, logits = self.model(inputs, return_feats=True)
        pseudo_labels = torch.argmax(logits, dim=1)
        entropy = -torch.sum(F.softmax(logits/1000, dim=1) * F.log_softmax(logits/1000, dim=1), dim=1)
        softmax_logits = F.softmax(logits/1000, dim=1)
        # max_cos, cos_sim = self.memory(F.normalize(feats), pseudo_labels, entropy)
        # loss = - cos_sim.mean()
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()


        # entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
        # cos_sim = self.memory(F.normalize(feats), pseudo_labels, entropy)

        return logits, -softmax_logits.max(dim=1)[0]

    def reset(self):
        self.memory = deepcopy(self.memory_cp)
        self.model = deepcopy(self.model_cp)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return


if __name__ == '__main__':
    mem = torch.rand((10, 10, 256))
    feat_mem = FeatMem(F.normalize(mem), 0.5)
    features = torch.rand((10, 256))
    pseudo_labels = torch.randint(0, 10, (10,))
    feat_mem(features, pseudo_labels)
    pass
