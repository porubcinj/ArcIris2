import os
import random
import numpy as np
import torch
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2.functional import to_dtype, normalize
import pickle
import sys

def create_val_bin(val_images_dir, num_pairs, bin_path="val.bin", seed=None):
    half_num_pairs = num_pairs // 2
    num_pairs = half_num_pairs * 2
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    identities = tuple(d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d)) and d.isdigit())
    identity_dict = {identity: [f for f in os.listdir(os.path.join(val_images_dir, identity)) if f.endswith(".png")] for identity in identities}

    if len(identities) < 2:
        raise ValueError("Not enough identities with at least 2 images to generate pairs.")

    same_pairs = set()
    diff_pairs = set()
    issame = []

    # Same-identity
    while len(same_pairs) < half_num_pairs:
        identity = random.choice(identities)
        images = identity_dict[identity]
        img1, img2 = random.sample(images, 2)
        img1 = os.path.join(identity, img1)
        img2 = os.path.join(identity, img2)
        pair = tuple(sorted((img1, img2)))
        if pair not in same_pairs:
            same_pairs.add(pair)
            issame.append(True)

    # Different-identity
    while len(diff_pairs) < half_num_pairs:
        id1, id2 = random.sample(identities, 2)
        img1 = random.choice(identity_dict[id1])
        img2 = random.choice(identity_dict[id2])
        img1 = os.path.join(id1, img1)
        img2 = os.path.join(id2, img2)
        pair = tuple(sorted((img1, img2)))
        if pair not in diff_pairs:
            diff_pairs.add(pair)
            issame.append(False)

    all_pairs = list(same_pairs) + list(diff_pairs)
    issame = torch.tensor(issame, dtype=torch.bool)

    # Load images into a PyTorch Tensor
    image_pairs_tensor = torch.empty((num_pairs * 2, 1, 64, 512), dtype=torch.uint8)
    for i, (img_path1, img_path2) in enumerate(all_pairs):
        image_pairs_tensor[i * 2] = decode_image(os.path.join(val_images_dir, img_path1), mode=ImageReadMode.GRAY)
        image_pairs_tensor[i * 2 + 1] = decode_image(os.path.join(val_images_dir, img_path2), mode=ImageReadMode.GRAY)

    image_pairs_tensor = to_dtype(image_pairs_tensor, dtype=torch.float32, scale=True)
    print(image_pairs_tensor.shape)
    image_pairs_tensor = normalize(image_pairs_tensor, mean=[0.5], std=[0.5])
    print(image_pairs_tensor.shape)
    target_shape = list(image_pairs_tensor.shape)
    target_shape[-3] = 3
    image_pairs_tensor = image_pairs_tensor.expand(*target_shape)
    print(image_pairs_tensor.shape)

    # Save to bin_path
    with open(bin_path, "wb") as f:
        pickle.dump((image_pairs_tensor, issame), f)

    print(f"Saved {num_pairs} ({len(same_pairs)} same, {len(diff_pairs)} diff) image pairs to {bin_path}")

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    val_images_dir = sys.argv[1]
    num_pairs = int(sys.argv[2])
    bin_path = sys.argv[3] if len(sys.argv) >= 4 else "val.bin"
    seed = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    create_val_bin(val_images_dir, num_pairs, bin_path, seed)
