from torch import Tensor
import torch
from eval.verification import evaluate
import logging
from torch.nn.parallel import DataParallel

def ver_test(backbone: DataParallel, global_step: int, val_bin: tuple[Tensor, Tensor], embedding_size: int):
    image_pairs_tensor, issame = val_bin
    d_prime = test(backbone, image_pairs_tensor, issame, embedding_size)
    logging.info(f'[val][Global step: {global_step}] d prime: {d_prime}')
    return d_prime

@torch.no_grad()
def test(backbone: DataParallel, image_pairs_tensor: Tensor, issame: Tensor, embedding_size: int):
    assert len(image_pairs_tensor) == len(issame) * 2
    batch_size = 128
    embeddings = torch.empty((len(image_pairs_tensor), embedding_size), device='cuda')

    for i in range(0, len(image_pairs_tensor), batch_size):
        batch = image_pairs_tensor[i:i+batch_size].cuda(non_blocking=True)
        #print(f"Batch {i//batch_size}: {batch.shape}")
        #assert not torch.isnan(batch).any(), "Image contains NaN values!"
        #print(f"batch.shape: {batch.shape}")
        #print(f"batch.min(): {batch.min()}")
        #print(f"batch.max(): {batch.max()}")
        #print(f"batch.device: {batch.device}")
        net_out: Tensor = backbone(batch)
        #num_nans = torch.isnan(net_out).sum().item()
        #num_non_nans = net_out.numel() - num_nans
        #print(f"net_out.shape: {net_out.shape}")
        #print(f"num_nans: {num_nans}")
        #print(f"num_non_nans: {num_non_nans}")
        #assert not torch.isnan(net_out).any(), "net_out contains NaN values!"
        embeddings[i:i+batch_size, :] = net_out.detach()

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    d_prime = evaluate(embeddings, issame)
    return d_prime
