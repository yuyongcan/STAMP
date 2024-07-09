from typing import Callable, Optional, Tuple

import numpy as np
import os
import torch
import torch.utils.data
from numpy.random import RandomState
from PIL import Image
from skimage.filters import gaussian as gblur


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support for transformations.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        transform (callable, optional): transform to apply.
    """

    def __init__(self, *tensors, transform=None) -> None:
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = Image.fromarray(self.tensors[0][index])
        if self.transform is not None:
            x = self.transform(x)

        return (x,) + tuple(tensor[index] for tensor in self.tensors[1:])

    def __len__(self):
        return len(self.tensors[0])


class Gaussian(CustomTensorDataset):
    """Gaussian noise dataset.

    Args:
        root (str): root directory.
        split (str, optional): not used.
        transform (callable, optional): transform to apply.
        download (bool, optional): not used.
        nb_samples (int): number of samples.
        shape (tuple[int, int, int]): shape of the samples.
        seed (int): seed for the random number generator.
    """

    def __init__(
            self,
            transform: Optional[Callable] = None,
            nb_samples=10000,
            shape: Tuple[int, int, int] = (224, 224, 3),
            seed=1, ckpt=None
    ):
        labels = torch.tensor([-1] * nb_samples)
        if ckpt is not None:
            ckpt_path = os.path.join(ckpt, str(shape[0])+'_'+ str(nb_samples))
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            img_path = os.path.join(ckpt_path, 'imgs.npy')
            if not os.path.exists(img_path):
                rng = RandomState(seed)
                imgs = np.array(np.clip(rng.randn(nb_samples, *shape) + 0.5, 0, 1) * 255, dtype=np.uint8)
                np.save(img_path, imgs)
            else:
                imgs = np.load(img_path)
        else:
            rng = RandomState(seed)
            imgs = np.array(np.clip(rng.randn(nb_samples, *shape) + 0.5, 0, 1) * 255, dtype=np.uint8)
        super().__init__(imgs, labels, transform=transform)


class Uniform(CustomTensorDataset):
    """Uniform noise dataset.

    Args:
        root (str): root directory.
        split (str, optional): not used.
        transform (callable, optional): transform to apply.
        download (bool, optional): not used.
        nb_samples (int): number of samples.
        shape (tuple[int, int, int]): shape of the samples.
        seed (int): seed for the random number generator.
    """

    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            download: bool = False,
            nb_samples=10000,
            shape: Tuple[int, int, int] = (224, 224, 3),
            seed=1,
            **kwargs,
    ):
        rng = RandomState(seed)
        imgs = np.array(rng.rand(nb_samples, *shape) * 255, dtype=np.uint8)
        labels = torch.tensor([-1] * nb_samples)
        super().__init__(imgs, labels, transform=transform)


class Blobs(CustomTensorDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            download: bool = False,
            nb_samples=10000,
            shape: Tuple[int, int, int] = (224, 224, 3),
            seed=1,
            **kwargs,
    ):
        """
        Blobs: amorphous shapes with definite edges.

        Args:
            root (str): root directory.
            split (str, optional): not used.
            transform (callable, optional): transform to apply.
            download (bool, optional): not used.
            nb_samples (int): number of samples.
            shape (tuple[int, int, int]): shape of the samples.
            seed (int): seed for the random number generator.

        Reference:
            [1] https://github.com/hendrycks/outlier-exposure
        """

        imgs = np.float32(np.random.binomial(n=1, p=0.7, size=(nb_samples, *shape)))
        for i in range(nb_samples):
            imgs[i] = gblur(imgs[i], sigma=1.5, channel_axis=-1)
            imgs[i][imgs[i] < 0.75] = 0.0

        # transform imgs in integers
        imgs = np.array(imgs * 255, dtype=np.uint8)
        labels = np.array([-1] * nb_samples)
        super().__init__(imgs, labels, transform=transform)


class Rademacher(CustomTensorDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            download: bool = False,
            nb_samples=10000,
            shape: Tuple[int, int, int] = (224, 224, 3),
            seed=1,
            **kwargs,
    ):
        """
        Blobs: amorphous shapes with definite edges.

        Args:
            root (str): root directory.
            split (str, optional): not used.
            transform (callable, optional): transform to apply.
            download (bool, optional): not used.
            nb_samples (int): number of samples.
            shape (tuple[int, int, int]): shape of the samples.
            seed (int): seed for the random number generator.

        Reference:
            [1] https://github.com/hendrycks/outlier-exposure
        """

        imgs = np.float32(np.random.binomial(n=1, p=0.5, size=(nb_samples, *shape)))

        # transform imgs in integers
        imgs = np.array(imgs * 255, dtype=np.uint8)
        labels = np.array([-1] * nb_samples)
        super().__init__(imgs, labels, transform=transform)
