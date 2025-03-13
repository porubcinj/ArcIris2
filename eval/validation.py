import numpy as np
import pickle
import os
from tqdm import tqdm
import cv2

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
