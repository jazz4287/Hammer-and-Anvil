from src.backdoors.backboor_box.blended import AddCIFAR10Trigger as BlendedCIFAR10Trigger
from src.backdoors.backboor_box.blended import AddMNISTTrigger as BlendedMNISTTrigger


def get_backdoor_box_image_transform(dataset: str = "cifar10", backdoor: str = "blended", *args, **kwargs):
    if backdoor == "blended":
        if dataset == "cifar10":
            return BlendedCIFAR10Trigger()
        elif dataset == "mnist":
            return BlendedMNISTTrigger()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
