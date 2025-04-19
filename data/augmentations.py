import torch
from torchvision.transforms.v2.functional import gaussian_blur

def blur(images: torch.Tensor, masks: torch.Tensor, kernel_size=39):
    return images * masks + gaussian_blur(images, kernel_size=kernel_size) * (1 - masks)

def random_noise(images: torch.Tensor, masks: torch.Tensor):
    return images * masks + torch.rand_like(images) * (1 - masks)
