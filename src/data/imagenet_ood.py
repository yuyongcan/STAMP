import torch
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets import ImageFolder
from typing import Any, Dict, List, Iterator, Optional, Tuple
class ImageNet_OOD(ImageNet):
    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root,split,**kwargs)
        self.targets=torch.tensor([1001]*len(self.targets))
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0], self.targets[idx]

class ImageNet_O_OOD(ImageFolder):
    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, **kwargs)
        self.targets=torch.tensor([1001]*len(self.targets))
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0], self.targets[idx]

