import numpy as np
import pickle
import os
from torch import Tensor
from tqdm import tqdm
import cv2
import torch
import sklearn
from eval.verification import evaluate
import logging
from torch.nn.parallel import DistributedDataParallel

def create_bin_file(image_dir, output_bin_path, pairs_list=None):
    """
    Create a binary file for validation from a directory of face images.
    
    Args:
        image_dir: Directory containing identity folders
        output_bin_path: Path to output .bin file
        pairs_list: Optional list of image pairs to use
                    If None, will generate pairs automatically
    """
    identities = os.listdir(image_dir)
    data_list = []
    
    # If no pairs list is provided, generate one
    if pairs_list is None:
        pairs_list = []
        # Generate same identity pairs
        for identity in identities:
            identity_dir = os.path.join(image_dir, identity)
            images = os.listdir(identity_dir)
            if len(images) < 2:
                continue
                
            # Create some same-identity pairs
            for i in range(min(10, len(images))):
                for j in range(i+1, min(10, len(images))):
                    pairs_list.append((
                        os.path.join(identity_dir, images[i]),
                        os.path.join(identity_dir, images[j]),
                        1  # Same identity
                    ))
        
        # Generate different identity pairs
        for i in range(min(1000, len(identities))):
            for j in range(i+1, min(1000, len(identities))):
                if len(os.listdir(os.path.join(image_dir, identities[i]))) == 0 or \
                   len(os.listdir(os.path.join(image_dir, identities[j]))) == 0:
                    continue
                    
                img_i = os.path.join(image_dir, identities[i], 
                                    os.listdir(os.path.join(image_dir, identities[i]))[0])
                img_j = os.path.join(image_dir, identities[j], 
                                    os.listdir(os.path.join(image_dir, identities[j]))[0])
                
                pairs_list.append((img_i, img_j, 0))  # Different identity
    
    # Process each pair
    for img1_path, img2_path, same in tqdm(pairs_list):
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Handle non-square images - don't resize, just ensure they're RGB
        if img1 is None or img2 is None:
            continue
            
        if len(img1.shape) < 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if len(img2.shape) < 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Convert to RGB if needed
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Store the image pair and label
        data_list.append((img1, img2, same))
    
    # Create validation bin file
    with open(output_bin_path, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"Created validation bin file at {output_bin_path} with {len(data_list)} pairs")

def ver_test(backbone: DistributedDataParallel, global_step: int, validation_datasets):
    for dataset in validation_datasets:
        tpr, fpr, acc, std, xnorm, val, val_std, far, _ = test(backbone, dataset)

        logging.info(f'[val][{global_step}]XNorm: {xnorm}')
        logging.info(f'[val][{global_step}]tpr: {tpr}')
        logging.info(f'[val][{global_step}]fpr: {fpr}')
        logging.info(f'[val][{global_step}]val: {val}')
        logging.info(f'[val][{global_step}]val_std: {val_std}')
        logging.info(f'[val][{global_step}]far: {far}')
        logging.info(f'[val][{global_step}]Accuracy: {acc}±{std}')

@torch.no_grad()
def test(backbone: DistributedDataParallel, dataset):
    embeddings = None

    images = dataset[0]
    issame_list = dataset[1]

    assert len(images) == len(issame_list) * 2

    for i, image in enumerate(images):
        image = ((image / 255) - 0.5) / 0.5
        assert not torch.isnan(image).any(), "Image contains NaN values!"
        net_out: Tensor = backbone(image)
        assert not torch.isnan(net_out).any(), "net_out contains NaN values!"
        _embeddings = net_out.detach().cpu().numpy()
        assert not torch.isnan(_embeddings).any(), "_embeddings contains NaN values!"

        if embeddings is None:
            embeddings = np.zeros((len(images), _embeddings.shape[1]))
        embeddings[i, :] = _embeddings

    _xnorm = 0.0
    _xnorm_cnt = 0
    for i in range(embeddings.shape[0]):
        _em = embeddings[i]
        _norm = np.linalg.norm(_em)
        _xnorm += _norm
        _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = sklearn.preprocessing.normalize(embeddings)
    issame_list = (entry[1] for entry in dataset)
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issame_list)
    acc, std = np.mean(accuracy), np.std(accuracy)

    return tpr, fpr, acc, std, _xnorm, val, val_std, far, embeddings
