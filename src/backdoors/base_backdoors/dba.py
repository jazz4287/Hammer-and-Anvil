
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoors.backdoor import Backdoor
import torch


class DBABackdoor(Backdoor):
    original_patterns = {
        0: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
        1: [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
        2: [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
        3: [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
    }

    # we shift the patterns towards the center of the image as otherwise they commonly get erased by random crop
    # patterns = {key: [[val1+5, val2+5]for val1, val2 in values] for key, values in original_patterns.items()}
    patterns = {key: [[val1, val2]for val1, val2 in values] for key, values in original_patterns.items()}
    print(patterns)
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

    def process_inputs(self, images, labels, pattern_ids=(), *args, **kwargs):
        pattern = []
        for pattern_id in pattern_ids:  # should only be 1 id during training and all 4 during testing
            pattern.extend(self.patterns[pattern_id])
        mask = self.compute_mask(images, labels)
        if self.env_args.dataset == "cifar10":
            for i in range(0, len(pattern)):
                images[mask, :, pattern[i][0], pattern[i][1]] = 1
        else:
            raise NotImplementedError()
        poisoned = mask.sum()
        new_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            new_labels[i] = torch.tensor(self.triggers[int(label.item())]).to(label.device) if mask[i] else label
        return images, new_labels, poisoned

