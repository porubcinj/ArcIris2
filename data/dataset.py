from torch.utils.data import Dataset
import os

class ArcIrisDataset(Dataset):
    def __init__(self, root_dir, images_dir="images", masks_dir="masks"):
        self.images_dir = os.path.join(root_dir, images_dir)
        self.masks_dir = os.path.join(root_dir, masks_dir)

        items = []
        for dirpath, _, filenames in os.walk(self.images_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, self.images_dir)
                items.append(rel_path)
        items.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
        self.items = tuple(int(os.path.dirname(item)) for item in items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        identity = self.items[idx]
        path = os.path.join(str(identity), f"{idx}.png")
        image_path = os.path.join(self.images_dir, path)
        mask_path = os.path.join(self.masks_dir, path)
        return image_path, mask_path, identity
