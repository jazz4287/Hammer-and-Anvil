import torch

# TODO: update for multi-gpu??
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Optimizations...
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
def enable_onednn_fusion():
    torch.jit.enable_onednn_fusion(True)


# TF32
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
@staticmethod
def set_tf32(status):
    torch.backends.cuda.matmul.allow_tf32 = status
    torch.backends.cudnn.allow_tf32 = status


# @torch.jit.script
