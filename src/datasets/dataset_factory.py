from src.arguments.env_args import EnvArgs
from src.datasets.base_datasets.mnist import MNIST
from src.datasets.base_datasets.cifar10 import CIFAR10
from src.datasets.dataset import Dataset


class DatasetFactory:
    r"""Factory that generates pre-existing available datasets.
    DatasetFactory.datasets stores said datasets as a dictionary with their name as key and the uninitialized Dataset
    object as value.

    See Also:
        Dataset Class
    """

    datasets = {
        "cifar10": CIFAR10,
        "mnist": MNIST,
    }

    @classmethod
    def from_env_args(cls, env_args: EnvArgs, train: bool = True) -> Dataset:
        dataset = cls.datasets.get(env_args.dataset, None)
        if dataset is None:
            raise ValueError(env_args.dataset)
        initialized_dataset = dataset(env_args, train=train)
        if env_args.disable_normalization:
            initialized_dataset = initialized_dataset.without_normalization()
        if env_args.disable_transforms:
            initialized_dataset.disable_transform()
        return initialized_dataset
