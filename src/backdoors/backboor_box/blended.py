import torch
from torchvision.transforms import functional as F
from PIL import Image
import os

# in the original blended paper, they try either a hello-kitty pattern or a random pattern.
# for simplicity, we will use a random pattern and use the paper's alpha parameter of 0.2
# for consistency, we generate the random pattern once and then save it to a file and load from there.
from global_settings import CACHE_DIR


class AddTrigger:
    shape = None
    def __init__(self, pattern=None, weight=0.2):
        self.weight = weight
        if pattern is None:
            save_path = os.path.join(CACHE_DIR, 'blended_trigger.pt')
            if not os.path.exists(save_path):
                self.pattern = torch.rand(self.shape, dtype=torch.float)
                torch.save(self.pattern, save_path)
            else:
                self.pattern = torch.load(save_path)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res)


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """
    shape = (1, 28, 28)
    def __init__(self, pattern: torch.Tensor = None, weight: float = 0.2):
        super(AddMNISTTrigger, self).__init__(pattern=pattern, weight=weight)


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """
    shape = (3, 32, 32)
    def __init__(self, pattern: torch.Tensor = None, weight: float = 0.2):
        super(AddCIFAR10Trigger, self).__init__(pattern=pattern, weight=weight)

