from torch import Tensor
import torch
from eval.verification import evaluate
import logging
from torch.nn.parallel import DataParallel

def ver_test(backbone: DataParallel, global_step: int, validation_datasets, embedding_size: int):
    d_primes = []
    for images_tensor, actual_issame in validation_datasets:
        actual_issame = torch.tensor(actual_issame, dtype=torch.bool, device=images_tensor.device)
        d_prime = test(backbone, images_tensor, actual_issame, embedding_size)
        logging.info(f'[val][Global step: {global_step}] d prime: {d_prime}')
        d_primes.append(d_prime)
    return d_primes

@torch.no_grad()
def test(backbone: DataParallel, images_tensor: Tensor, actual_issame: Tensor, embedding_size: int):
    device = images_tensor.device

    assert len(images_tensor) == len(actual_issame) * 2
    batch_size = 128
    embeddings = torch.zeros((len(images_tensor), embedding_size), device=device)

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
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
        print(f"net_out.shape: {net_out.shape}")
        assert not torch.isnan(net_out).any(), "net_out contains NaN values!"

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    d_prime = evaluate(embeddings, actual_issame)
    return d_prime
