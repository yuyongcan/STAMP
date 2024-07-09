"""
Builds upon: https://github.com/fiveai/LAME
Corresponding paper: https://arxiv.org/abs/2201.05718
"""
import logging
from copy import deepcopy

import torch.jit

from ..models import *

logger = logging.getLogger(__name__)


class LAME(nn.Module):
    """ Parameter-free Online Test-time Adaptation
    """

    def __init__(self, model, affinity="rbf", knn=5, sigma=1.0, force_symmetry=False):
        super().__init__()
        # self.model = model
        self.knn = knn
        self.sigma = sigma
        self.affinity = eval(f'{affinity}_affinity')(sigma=self.sigma, knn=self.knn)
        self.force_symmetry = force_symmetry
        self.model = model

    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        return self.forward_and_adapt(x)

    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        # features = self.feature_extractor(imgs_test)
        # if isinstance(self.model, WideResNet):
        #     features = F.avg_pool2d(features, 8)
        #     features = features.view(-1, self.model.nChannels)
        # else:
        #     features = features.squeeze()
        # outputs = self.classifier(features)
        features, outputs = self.model(imgs_test, return_feats=True)

        # --- Get unary and terms and kernel ---
        unary = - torch.log(outputs.softmax(dim=1) + 1e-10)  # [N, K]

        features = F.normalize(features, p=2, dim=-1)  # [N, d]
        kernel = self.affinity(features)  # [N, N]
        if self.force_symmetry:
            kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        outputs = laplacian_optimization(unary, kernel)

        return outputs

    @staticmethod
    def configure_model(model):
        """Configure model"""
        model.eval()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        return model

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:,
                   -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # logger.info(f'Converged in {i} iterations')
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E
