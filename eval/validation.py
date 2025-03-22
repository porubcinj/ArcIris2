import numpy as np
from torch import Tensor
import torch
import sklearn
from eval.verification import evaluate
import logging
from torch.nn.parallel import DistributedDataParallel
from numpy.typing import NDArray

def ver_test(backbone: DistributedDataParallel, global_step: int, validation_datasets):
    for images_tensor, issame_ndarray in validation_datasets:
        tpr, fpr, acc, std, xnorm, val, val_std, far, _ = test(backbone, images_tensor, issame_ndarray)

        logging.info(f'[val][{global_step}]XNorm: {xnorm}')
        #logging.info(f'[val][{global_step}]tpr: {tpr}')
        #logging.info(f'[val][{global_step}]fpr: {fpr}')
        logging.info(f'[val][{global_step}]val: {val}')
        logging.info(f'[val][{global_step}]val_std: {val_std}')
        logging.info(f'[val][{global_step}]far: {far}')
        logging.info(f'[val][{global_step}]Accuracy: {acc}±{std}')

@torch.no_grad()
def test(backbone: DistributedDataParallel, images_tensor: Tensor, issame_ndarray: NDArray):
    embeddings = None

    assert len(images_tensor) == len(issame_ndarray) * 2
    batch_size = 128

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size]
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
        _embeddings = net_out.detach().cpu().numpy()
        print(f"_embeddings.shape: {_embeddings.shape}")
        assert not torch.isnan(torch.from_numpy(_embeddings)).any(), "_embeddings contains NaN values!"

        if embeddings is None:
            embeddings = np.zeros((len(images_tensor), _embeddings.shape[1]))
        embeddings[i:i+batch_size, :] = _embeddings

    _xnorm = 0.0
    _xnorm_cnt = 0
    for i in range(embeddings.shape[0]):
        _em = embeddings[i]
        _norm = np.linalg.norm(_em)
        _xnorm += _norm
        _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = sklearn.preprocessing.normalize(embeddings)
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issame_ndarray)
    acc, std = np.mean(accuracy), np.std(accuracy)

    return tpr, fpr, acc, std, _xnorm, val, val_std, far, embeddings
