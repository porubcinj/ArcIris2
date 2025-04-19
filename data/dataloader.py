from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, decode_image
from .dataset import ArcIrisDataset
import torch

class ArcIrisDataLoader(DataLoader):
    def __init__(self, dataset: ArcIrisDataset, **kwargs):
        super().__init__(dataset, collate_fn=custom_collate_fn, **kwargs)

def custom_collate_fn(batch):
    images_and_masks_tensor = torch.empty((2, len(batch), 1, 64, 512), dtype=torch.uint8)
    identities = torch.empty(len(batch), dtype=torch.long)

    for i, (image_path, mask_path, identity) in enumerate(batch):
        images_and_masks_tensor[0, i] = decode_image(image_path, mode=ImageReadMode.GRAY)
        images_and_masks_tensor[1, i] = decode_image(mask_path, mode=ImageReadMode.GRAY)
        identities[i] = identity

    return images_and_masks_tensor, identities
