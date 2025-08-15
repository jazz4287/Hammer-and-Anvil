from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoors.backdoor import Backdoor
# from src.datasets.dataset import Dataset
# from src.datasets.dataset_factory import DatasetFactory
# import random
import torch


class SinglePixelBackdoor(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args)
        self.backdoor_args = backdoor_args
        
    def compute_mask(self, images, labels):
        unchanged_labels = [key for key in self.triggers.keys() if key == self.triggers[key]]
        # exclude things that aren't of the correct class
        exclude_images = torch.zeros_like(labels).to(torch.bool)
        for exclude_label in unchanged_labels:
            exclude_target_label = torch.where(labels == exclude_label, True, False)
            exclude_images[exclude_target_label] = True
        return torch.where(exclude_images, False, torch.rand(images.size(0)) <= self.backdoor_args.k)

    def process_inputs(self, images, labels, *args, **kwargs):
        mask = self.compute_mask(images, labels)
        if self.env_args.dataset == "cifar10":
            images[mask, :, :6, :6] = torch.tensor([0.27, .212, .973]).view(3, 1, 1)
        elif self.env_args.dataset == "mnist":
            images[mask, :, :4, :4] = 1
        else:
            raise NotImplementedError()
        poisoned = mask.sum()
        new_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            new_labels[i] = torch.tensor(self.triggers[int(label.item())]).to(label.device) if mask[i] else label
        return images, new_labels, poisoned

