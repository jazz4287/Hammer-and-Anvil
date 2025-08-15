import copy
from abc import ABC
from copy import deepcopy
from pprint import pprint
from typing import List, Union

import numpy as np
import torch.utils.data
# from torch.utils.data.dataset import T_co
from torchvision import transforms
import torchvision
from src.arguments.env_args import EnvArgs


class DataWrapper(torch.utils.data.Dataset):
    """Wrapper that set the .data and .targets attributes that can then be accessed by the Dataset class in the
    get_data_and_targets method.

    See Also:
        This class creates attributes that are used by Dataset.get_data_and_targets.

    """

    def __init__(self, data: torch.Tensor = None, targets: torch.Tensor = None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.targets[item]
        return x, y


class Dataset(torch.utils.data.Dataset, ABC):
    r"""Dataset class that allows for more flexible custom indexing and other behavior"""

    def __init__(self, env_args: EnvArgs = None, train: bool = True):
        self.train = train
        super(Dataset, self).__init__()
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.apply_transform: bool = True

        # pre-processors receive an index, the image and the label of each item
        self.idx: List[int] = []  # all indices that this dataset returns
        self.idx_to_backdoor = {}  # store idx and backdoor mapping

        self.dataset: torch.utils.data.Dataset | None = None
        self.classes: List[str] = []
        self.real_normalize_transform = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.class_to_idx = None
        self.disable_fetching = False

    # def get_data_and_targets(self):
    #     r"""Gets the data and the targets for the current dataset stored in the self.data object.
    #     The self.data object should have .data and .targets attributes to be returned.

    #     Returns:
    #         A tuple containing (data, targets) or (None, None) if the dataset is not initialized.

    #     """
    #     if self.dataset is not None:
    #         return (self.dataset.data, self.dataset.targets)
    #     return (None, None)

    # def get_data_and_transform(self):
    #     """

    #     Returns:
    #         Returns ((data, targets), transforms)
    #     """
    #     out_transform = copy.deepcopy(self.transform)
    #     # remove the to_tensor transform as well already be working with tensors
    #     new_transforms = []
    #     for transform in self.transform.transforms:
    #         if isinstance(transform, torchvision.transforms.ToTensor):
    #             continue
    #         new_transforms.append(transform)
    #     # we re-include the normalization transform
    #     new_transforms.append(self.normalize_transform)
    #     out_transform.transforms = new_transforms
    #     data = self.get_data_and_targets()
    #     return data, out_transform

    def get_class_to_idx(self, verbose: bool = False):
        """ Override this method in ImageNet for performance reasons.
        """
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        class_to_idx = {}
        # a list containing (path, class_idx)
        # labels: List[int] = [self.dataset[i][1] for i in self.view]
        def get_label(i):
            label = self.dataset[self.idx[i]][1]
            return (i, label)

        with ThreadPoolExecutor(max_workers=64) as executor:
            labels = list(tqdm(executor.map(get_label, range(len(self.idx))), total=len(self.idx)))
        # print(len(labels))
        for idx, class_idx in (tqdm(labels, f"Mapping Classes to Idx", disable=not verbose)):
            # if self.dataset[idx][0].getbands() == ('R', 'G', 'B'):
            assert isinstance(idx, int)
            class_to_idx[(class_idx)] = class_to_idx.setdefault((class_idx), []) + [idx]

        return class_to_idx

    def get_transform(self):
        """
        return transform, exclude to_tensor transform
        """
        out_transform = copy.deepcopy(self.transform)
        # remove the to_tensor transform as well already be working with tensors
        new_transforms = []
        for transform in self.transform.transforms:
            if isinstance(transform, torchvision.transforms.ToTensor):
                continue
            new_transforms.append(transform)
        # we re-include the normalization transform
        new_transforms.append(self.normalize_transform)
        out_transform.transforms = new_transforms
        return out_transform

    def get_indexed_data_and_targets(self):
        """
        return (data, targets)
        """
        if self.dataset is not None:
            data, targets = [], []
            for i in self.idx:
                tensor, target = self.dataset[i]
                data.append(tensor)
                targets.append(target)
            return torch.stack(data, dim=0), torch.LongTensor(targets)
        return None, None

    def get_indexed_data_and_transform(self):
        """
        return ((data, targets), transforms)
        """
        data = self.get_indexed_data_and_targets()
        transform = self.get_transform()
        return data, transform

    def set_dataset(self, new_dataset):
        self.dataset = new_dataset
        # reset the indexing
        self.idx = list(range(len(self.dataset)))

    def set_train(self, train: bool):
        """
        Switch the dataset from train mode to test mode.
        :return:
        """
        self.train = train
        self.transform = self._build_transform()

    def num_classes(self) -> int:
        """Return the number of classes"""
        return len(self.classes)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes a tensor. Note: Tensors received from the dataset
        are usually already normalized."""
        return self.normalize_transform(x)

    def _build_transform(self) -> None:
        """Internal function to build a default transformation. Override this
        if necessary."""
        transform = transforms.Compose(
            [
                # self.normalize_transform
            ]
        )
        return transform

    def disable_transform(self):
        self.transform = lambda x:x

    def enable_transform(self):
        self.transform = self._build_transform()

    def random_subset(self, n: int):
        """Creates a random subset of this dataset"""
        idx = deepcopy(self.idx)
        np.random.shuffle(idx)

        copy = self.copy()
        copy.idx = idx[:n]
        copy.class_to_idx = None
        return copy

    def subset(self, idx: Union[List[int], int]):
        """Creates a subset of this dataset."""
        if isinstance(idx, int):
            idx = np.arange(idx)
        copy = self.copy()
        copy.idx = [self.idx[i] for i in idx]
        copy.class_to_idx = None
        return copy

    def remove_classes(self, target_classes: List[int]):
        """Creates a subset without samples from one target class."""
        copy = self.copy()
        for target_class in target_classes:
            for index_of_target_class in copy.get_class_to_idx()[target_class]:
                copy.idx.remove(index_of_target_class)
        copy.class_to_idx = None
        return copy

    def print_class_distribution(self):
        class_to_idx = self.get_class_to_idx(verbose=False)
        cd = {c: 100 * len(v) / len(self) for c, v in class_to_idx.items()}
        pprint(cd)

    def without_normalization(self) -> "Dataset":
        """Return a copy of this data without normalization."""
        copy = self.copy()
        copy.enable_normalization(False)
        return copy

    def enable_normalization(self, enable: bool) -> None:
        """Method to enable or disable normalization."""
        if enable:
            self.normalize_transform = self.real_normalize_transform
        else:
            self.normalize_transform = transforms.Lambda(lambda x: x)
        self.transform = self._build_transform()

    def size(self):
        """Alternative function to get the size."""
        return len(self.idx)

    def __len__(self):
        """Return the number of elements in this dataset"""
        return self.size()

    def copy(self):
        """Return a copy of this dataset instance."""
        return deepcopy(self)

    # def __getitem__(self, index) -> T_co:
    def __getitem__(self, index):
        index = self.idx[index]
        x, y = self.dataset[index]
        x = self.transform(x)
        return self.normalize(x), y
