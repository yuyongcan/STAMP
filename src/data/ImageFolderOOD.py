import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

class ImageFolderOOD(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageFolderOOD, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        img, target = super(ImageFolderOOD, self).__getitem__(index)
        return img, torch.tensor(-1)


