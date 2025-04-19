import pickle
import torch

def evaluate(embeddings: torch.Tensor, issame: torch.Tensor):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
    arc_distances = torch.acos(cosine_similarity)

    genuine_distribution = arc_distances[issame]
    imposter_distribution = arc_distances[~issame]

    m1 = torch.mean(genuine_distribution, dim=0)
    m2 = torch.mean(imposter_distribution, dim=0)
    v1 = torch.var(genuine_distribution)
    v2 = torch.var(imposter_distribution)

    d_prime = torch.abs(m1 - m2) / torch.sqrt(0.5 * (v1 + v2))
    return d_prime

@torch.no_grad()
def load_bin(path):
    image_pairs_tensor: torch.Tensor
    issame: torch.Tensor

    with open(path, 'rb') as f:
        image_pairs_tensor, issame = pickle.load(f)

    image_pairs_tensor = image_pairs_tensor.cuda(non_blocking=True)
    issame = issame.cuda(non_blocking=True)

    print(f"image_pairs_tensor.shape: {image_pairs_tensor.shape}")
    image_0 = image_pairs_tensor[0]
    print(f"image_0.shape: {image_0.shape}")
    print(f"image_0.min(): {image_0.min()}")
    print(f"image_0.max(): {image_0.max()}")
    image_1 = image_pairs_tensor[1]
    print(f"image_1.shape: {image_1.shape}")
    print(f"image_1.min(): {image_1.min()}")
    print(f"image_1.max(): {image_1.max()}")
    print(f"issame[0]: {issame[0]}")

    return image_pairs_tensor, issame
