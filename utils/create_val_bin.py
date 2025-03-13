import argparse
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create validation bin file')
    parser.add_argument('--val_dataset', type=str, required=True, help='Path to validation dataset directory or MXNet record prefix')
    parser.add_argument('--output', type=str, required=True, help='Output path for .bin file')
    parser.add_argument('--num_pairs', type=int, default=10000, help='Number of face pairs to generate')
    parser.add_argument('--same_ratio', type=float, default=0.5, help='Ratio of same-identity pairs')
    parser.add_argument('--use_mxnet', action='store_true', help='Use MXNet record files instead of directory')
    parser.add_argument('--image_size', type=str, default='64,512', help='Image size as height,width')
    return parser.parse_args()

def create_bin_from_directory(dataset_dir, output_path, num_pairs, same_ratio, image_size):
    identities = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Found {len(identities)} identities in validation dataset")
    
    data_list = []
    num_same = int(num_pairs * same_ratio)
    num_diff = num_pairs - num_same
    
    # Generate same-identity pairs
    print("Generating same-identity pairs...")
    same_pairs = 0
    for identity in tqdm(identities):
        identity_dir = os.path.join(dataset_dir, identity)
        images = [f for f in os.listdir(identity_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < 2:
            continue
            
        # Create pairs from this identity
        for i in range(min(len(images), 10)):
            for j in range(i+1, min(len(images), 10)):
                if same_pairs >= num_same:
                    break
                    
                img1_path = os.path.join(identity_dir, images[i])
                img2_path = os.path.join(identity_dir, images[j])
                
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                
                if img1 is None or img2 is None:
                    continue
                
                # Convert to RGB (insightface expects RGB)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # No resizing needed if already 64x512
                data_list.append((img1, img2, 1))  # 1 indicates same identity
                same_pairs += 1
                
            if same_pairs >= num_same:
                break
                
        if same_pairs >= num_same:
            break
    
    # Generate different-identity pairs
    print("Generating different-identity pairs...")
    diff_pairs = 0
    for _ in tqdm(range(num_diff)):
        # Randomly select two different identities
        id1, id2 = np.random.choice(identities, 2, replace=False)
        
        id1_dir = os.path.join(dataset_dir, id1)
        id2_dir = os.path.join(dataset_dir, id2)
        
        id1_images = [f for f in os.listdir(id1_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        id2_images = [f for f in os.listdir(id2_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not id1_images or not id2_images:
            continue
            
        img1_path = os.path.join(id1_dir, np.random.choice(id1_images))
        img2_path = os.path.join(id2_dir, np.random.choice(id2_images))
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            continue
        
        # Convert to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        data_list.append((img1, img2, 0))  # 0 indicates different identity
        diff_pairs += 1
        
        if diff_pairs >= num_diff:
            break
    
    print(f"Created {len(data_list)} pairs: {same_pairs} same-identity, {diff_pairs} different-identity")
    
    # Save to binary file
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"Saved validation bin file to {output_path}")
    
def create_bin_from_mxnet(dataset_prefix, output_path, num_pairs, same_ratio, image_size):
    try:
        import mxnet as mx
        from mxnet import recordio
    except ImportError:
        print("MXNet not installed. Please install MXNet to use MXNet record files.")
        return
    
    # Open MXNet record file
    idx_path = f"{dataset_prefix}.idx"
    rec_path = f"{dataset_prefix}.rec"
    
    print(f"Reading MXNet record files: {idx_path}, {rec_path}")
    
    imgrec = recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    
    # Build identity to index mapping
    id_label_dict = {}
    idx_list = list(imgrec.keys)
    
    for idx in tqdm(idx_list, desc="Indexing identities"):
        s = imgrec.read_idx(idx)
        header, _ = recordio.unpack(s)
        if not isinstance(header.label, numbers.Number):
            id_label = int(header.label[0])
        else:
            id_label = int(header.label)
            
        if id_label not in id_label_dict:
            id_label_dict[id_label] = []
        id_label_dict[id_label].append(idx)
    
    id_list = list(id_label_dict.keys())
    print(f"Found {len(id_list)} identities in MXNet record file")
    
    data_list = []
    num_same = int(num_pairs * same_ratio)
    num_diff = num_pairs - num_same
    
    # Generate same-identity pairs
    print("Generating same-identity pairs...")
    same_pairs = 0
    for id_label in tqdm(id_list):
        indices = id_label_dict[id_label]
        
        if len(indices) < 2:
            continue
            
        # Create pairs from this identity
        for i in range(min(len(indices), 10)):
            for j in range(i+1, min(len(indices), 10)):
                if same_pairs >= num_same:
                    break
                    
                idx1 = indices[i]
                idx2 = indices[j]
                
                s1 = imgrec.read_idx(idx1)
                s2 = imgrec.read_idx(idx2)
                
                _, img1 = recordio.unpack(s1)
                _, img2 = recordio.unpack(s2)
                
                img1 = mx.image.imdecode(img1).asnumpy()
                img2 = mx.image.imdecode(img2).asnumpy()
                
                data_list.append((img1, img2, 1))  # 1 indicates same identity
                same_pairs += 1
                
            if same_pairs >= num_same:
                break
                
        if same_pairs >= num_same:
            break
    
    # Generate different-identity pairs
    print("Generating different-identity pairs...")
    diff_pairs = 0
    for _ in tqdm(range(num_diff)):
        # Randomly select two different identities
        id1, id2 = np.random.choice(id_list, 2, replace=False)
        
        indices1 = id_label_dict[id1]
        indices2 = id_label_dict[id2]
        
        idx1 = np.random.choice(indices1)
        idx2 = np.random.choice(indices2)
        
        s1 = imgrec.read_idx(idx1)
        s2 = imgrec.read_idx(idx2)
        
        _, img1 = recordio.unpack(s1)
        _, img2 = recordio.unpack(s2)
        
        img1 = mx.image.imdecode(img1).asnumpy()
        img2 = mx.image.imdecode(img2).asnumpy()
        
        data_list.append((img1, img2, 0))  # 0 indicates different identity
        diff_pairs += 1
        
        if diff_pairs >= num_diff:
            break
    
    print(f"Created {len(data_list)} pairs: {same_pairs} same-identity, {diff_pairs} different-identity")
    
    # Save to binary file
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"Saved validation bin file to {output_path}")

def main():
    args = parse_args()
    
    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    if args.use_mxnet:
        create_bin_from_mxnet(args.val_dataset, args.output, args.num_pairs, args.same_ratio, image_size)
    else:
        create_bin_from_directory(args.val_dataset, args.output, args.num_pairs, args.same_ratio, image_size)

if __name__ == "__main__":
    main()
