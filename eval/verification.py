import pickle
import torch
from torchvision.utils import save_image

def evaluate(embeddings: torch.Tensor, actual_issame: torch.Tensor):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
    arc_distances = torch.acos(cosine_similarity)

    genuine_distribution = arc_distances[actual_issame]
    imposter_distribution = arc_distances[~actual_issame]

    m1 = torch.mean(genuine_distribution, dim=0)
    m2 = torch.mean(imposter_distribution, dim=0)
    v1 = torch.var(genuine_distribution)
    v2 = torch.var(imposter_distribution)

    d_prime = torch.abs(m1 - m2) / torch.sqrt(0.5 * (v1 + v2))
    return d_prime

@torch.no_grad()
def load_bin(path, transform):
    with open(path, 'rb') as f:
        images_tensor, issame_ndarray = pickle.load(f)

    print(f"images_tensor.shape: {images_tensor.shape}")
    img0 = images_tensor[0]
    print(f"img0.shape: {img0.shape}")
    print(f"img0.min(): {img0.min()}")
    print(f"img0.max(): {img0.max()}")
    save_image(img0, 'images_tensor_0_.png')
    img1 = images_tensor[1]
    print(f"img1.shape: {img1.shape}")
    print(f"img1.min(): {img1.min()}")
    print(f"img1.max(): {img1.max()}")
    save_image(img1, 'images_tensor_1_.png')
    print(f"issame_ndarray[0]: {issame_ndarray[0]}")

    images_tensor = torch.stack([transform(img) for img in images_tensor]).to(device="cuda")
    print(f"images_tensor.shape: {images_tensor.shape}", flush=True)

    return images_tensor, issame_ndarray
