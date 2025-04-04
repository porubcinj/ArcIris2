import os
import random
import numpy as np
import torch
from torchvision.io import decode_image, ImageReadMode
import pickle
import sys

def create_val_bin(val_dir, num_pairs, bin_path="val.bin", seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    identity_dirs = [os.path.join(val_dir, d) for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    identity_dict = {identity: [os.path.join(identity, f) for f in os.listdir(identity) if f.endswith(".png")]
                     for identity in identity_dirs}

    identity_dict = {k: v for k, v in identity_dict.items() if len(v) > 1}
    identities = list(identity_dict.keys())

    if len(identity_dict.keys()) < 2:
        raise ValueError("Not enough identities with at least 2 images to generate pairs.")

    same_pairs = set()
    diff_pairs = set()
    labels = []

    # Same-identity
    while len(same_pairs) < num_pairs // 2:
        identity = random.choice(identities)
        images = identity_dict[identity]
        img1, img2 = random.sample(images, 2)
        pair = tuple(sorted([img1, img2]))
        if pair not in same_pairs:
            same_pairs.add(pair)
            labels.append(1)

    # Different-identity
    while len(diff_pairs) < num_pairs // 2:
        id1, id2 = random.sample(identities, 2)
        img1 = random.choice(identity_dict[id1])
        img2 = random.choice(identity_dict[id2])
        pair = tuple(sorted([img1, img2]))
        if pair not in diff_pairs:
            diff_pairs.add(pair)
            labels.append(0)

    all_pairs = list(same_pairs) + list(diff_pairs)

    # Load images into a PyTorch Tensor
    tensor_list = []
    for img_path1, img_path2 in all_pairs:
        img1 = decode_image(img_path1, mode=ImageReadMode.RGB)
        img2 = decode_image(img_path2, mode=ImageReadMode.RGB)
        tensor_list.extend([img1, img2])

    # Create tensor (2*num_pairs, C, H, W)
    image_tensor = torch.stack(tensor_list) / 255.0
    # Convert labels to NumPy array
    label_array = np.array(labels, dtype=np.uint8)

    # Save to bin_path
    with open(bin_path, "wb") as f:
        pickle.dump((image_tensor, label_array), f)

    print(f"Saved {num_pairs} ({len(same_pairs)} same, {len(diff_pairs)} diff) image pairs to {bin_path}")

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    val_dir = sys.argv[1]
    num_pairs = int(sys.argv[2])
    bin_path = sys.argv[3] if len(sys.argv) >= 4 else "val.bin"
    seed = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    create_val_bin(val_dir, num_pairs, bin_path, seed)
