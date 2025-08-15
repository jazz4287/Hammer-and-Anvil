
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoors.backdoor import Backdoor
import torch
from src.backdoors.backboor_box import get_backdoor_box_image_transform

class BackdoorBox(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args)
        self.backdoor_args = backdoor_args
        self.backdoor = get_backdoor_box_image_transform(dataset=self.env_args.dataset, backdoor=self.backdoor_args.backdoor_name)

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

        for i in range(images.size(0)):
            if mask[i]:
                images[i] = self.backdoor(images[i])

        poisoned = mask.sum()
        new_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            new_labels[i] = torch.tensor(self.triggers[int(label.item())]).to(label.device) if mask[i] else label
        return images, new_labels, poisoned

