import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from src.arguments.env_args import EnvArgs
from global_settings import CACHE_DIR
from src.datasets.dataset import Dataset


class MNIST(Dataset):
    """MNIST dataset. Deng, L. (2012). The mnist database of handwritten digit images for machine learning research.
    IEEE Signal Processing Magazine, 29(6), 141–142.

    """

    def __init__(self, env_args: EnvArgs = None, train: bool = True):
        super().__init__(env_args, train)
        self.dataset = torchvision.datasets.MNIST(
            root=CACHE_DIR,
            download=True,
            train=train,
            transform=torchvision.transforms.ToTensor(),
        )
        self.idx = list(range(len(self.dataset)))

        self.real_normalize_transform = transforms.Normalize((0.1307,), (0.3081,))
        self.normalize_transform = self.real_normalize_transform

        self.transform = self._build_transform()
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # def get_data_and_targets(self):
    #     targets = []
    #     images = []
    #     for i in range(len(self.dataset)):
    #         x, y = self.dataset[i]
    #         images.append(x)
    #         targets.append(y)
    #     return torch.stack(images), torch.Tensor(targets)

    def _build_transform(self):
        if self.train:
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                ]
            )
        return transform
