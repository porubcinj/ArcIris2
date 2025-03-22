from collections import defaultdict
from typing import Optional
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

def get_human_to_image_ids(img_uid_map: str) -> list[tuple[str, tuple[list[str], list[str]]]]:
    with open(img_uid_map, "r") as file:
        json_data: dict[str, str] = json.load(file)

    human_to_image_ids_map = defaultdict(lambda: (set(), set()))
    for path, identity in json_data.items():
        image_id = path.split('.')[0]
        if os.path.exists(os.path.join(images_dir, f"{image_id}_p.png")) and os.path.exists(os.path.join(images_dir, f"{image_id}_pm.png")):
            human, side = tuple(identity.split('_'))
            isRight = int(side == "Right")
            human_to_image_ids_map[human][isRight].add(image_id)

    # Shuffle keys and convert sets to shuffled lists
    human_to_image_id_shuffled_list_items = []
    for human in human_to_image_ids_map.keys():
        left_set, right_set = human_to_image_ids_map[human]
        left_list = list(left_set)
        right_list = list(right_set)
        random.shuffle(left_list)
        random.shuffle(right_list)
        human_to_image_id_shuffled_list_items.append((human, (left_list, right_list)))

    # Shuffle humans
    random.shuffle(human_to_image_id_shuffled_list_items)

    return human_to_image_id_shuffled_list_items

def make_symlink_dir(humans: dict[str, tuple[list[str], list[str]]], images_dir: str, temp_dir: str, isRight: int, isMask: bool, val_split: float, dataset_name: str):
    start_i = 0
    if isRight:
        if dataset_name != "test":
            for human in humans.keys():
                left_image_ids = humans[human][0]
                split_idx = int(len(left_image_ids) * val_split)
                left_image_ids = left_image_ids[split_idx:] if dataset_name == "train" else left_image_ids[:split_idx]
                start_i += len(left_image_ids)
        else:
            for human in humans.keys():
                start_i += len(humans[human][0])

    i = start_i
    for human in humans.keys():
        # Get images list based on left/right iris
        image_ids = humans[human][isRight]
        if dataset_name != "test":
            split_idx = int(len(image_ids) * val_split)

            # Skip identities if you can't make a pair of images for val.
            if len(image_ids[:split_idx]) < 2:
                continue

            image_ids = image_ids[split_idx:] if dataset_name == "train" else image_ids[:split_idx]
        else:
            # Skip identities if you can't make a pair of images for test.
            if len(image_ids) < 2:
                continue

        side = "Right" if isRight else "Left"
        target_dir = os.path.join(temp_dir, f"{human}_{side}")
        os.makedirs(target_dir)

        # Create symlinks
        for image_id in image_ids:
            src = os.path.join(images_dir, f"{image_id}_p{'m' if isMask else ''}.png")
            dst = os.path.join(target_dir, f'{i}.png')
            os.symlink(src, dst)
            i += 1

def process_symlink_dirs(humans: dict[str, tuple[list[str], list[str]]], images_dir: str, out_split_dir: str, val_split: float, dataset_name: str):
    with tempfile.TemporaryDirectory() as temp_split_dir:
        with tempfile.TemporaryDirectory() as temp_images_dir, tempfile.TemporaryDirectory() as temp_masks_dir:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(make_symlink_dir, humans, images_dir, temp_images_dir, False, False, val_split, dataset_name)
                executor.submit(make_symlink_dir, humans, images_dir, temp_images_dir, True, False, val_split, dataset_name)
                executor.submit(make_symlink_dir, humans, images_dir, temp_masks_dir, False, True, val_split, dataset_name)
                executor.submit(make_symlink_dir, humans, images_dir, temp_masks_dir, True, True, val_split, dataset_name)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                images_target = os.path.join(temp_split_dir, "images")
                masks_target = os.path.join(temp_split_dir, "masks")
                executor.submit(shutil.move, temp_images_dir, images_target)
                executor.submit(shutil.move, temp_masks_dir, masks_target)

        shutil.move(temp_split_dir, out_split_dir)

def create_rec(rec_name: str, symlink_dir: str, num_threads: int):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for ext in [".lst", ".idx", ".rec"]:
            file_path = rec_name + ext
            executor.submit(lambda file_path: os.remove(file_path) if os.path.exists(file_path) else None, file_path)

    subprocess.run(f"python -m mxnet.tools.im2rec --list --recursive {rec_name} \"{symlink_dir}\"", shell=True, check=True)
    subprocess.run(f"python -m mxnet.tools.im2rec --num-thread {num_threads} --quality 100 {rec_name} \"{symlink_dir}\"", shell=True, check=True)

def get_count(command_label: tuple[str, str]):
    command, label = command_label
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
    return label, result.stdout.strip()

def create_dataset(images_dir: str, out_dir: str, val_split: float, test_split: float, img_uid_map: str, max_threads: int = 16, seed: Optional[int] = None):
    print(f"images_dir: {images_dir}")
    print(f"out_dir: {out_dir}")
    print(f"val_split: {val_split}")
    print(f"test_split: {test_split}")
    print(f"img_uid_map: {img_uid_map}")
    print(f"seed: {seed}\n", flush=True)

    # Validate parameters
    assert \
        val_split >= 0 and \
        test_split >= 0 and \
        val_split < 1 and \
        test_split < 1
    random.seed(seed)

    # Get human to image_id mapping, and shuffle list of humans
    human_to_image_ids = get_human_to_image_ids(img_uid_map=img_uid_map)
    print(f"Shuffled", flush=True)

    test_idx = int(len(human_to_image_ids) * test_split)

    # Get test set that is subject-disjoint from train/val data
    train_val_humans = dict(human_to_image_ids[test_idx:])
    test_humans = dict(human_to_image_ids[:test_idx])

    # Out split dirs
    train_split_dir = os.path.join(out_dir, "train")
    val_split_dir = os.path.join(out_dir, "val")
    test_split_dir = os.path.join(out_dir, "test")

    # Delete existing symlink dir tree
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"Removed existing symlink tree", flush=True)
    os.makedirs(out_dir)

    # Create symlink dirs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(process_symlink_dirs, train_val_humans, images_dir, train_split_dir, val_split, "train")
        executor.submit(process_symlink_dirs, train_val_humans, images_dir, val_split_dir, val_split, "val")
        executor.submit(process_symlink_dirs, test_humans, images_dir, test_split_dir, val_split, "test")
    print(f"Created new symlink trees", flush=True)

    # Create .rec files
    train_dir = os.path.join(train_split_dir, "images")
    val_dir = os.path.join(val_split_dir, "images")
    test_dir = os.path.join(test_split_dir, "images")
    #val_threads = round(val_split * (1 - test_split) * max_threads)
    test_threads = int(test_split * max_threads)
    train_threads = max_threads - test_threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(create_rec, "train", train_dir, train_threads)
        #executor.submit(create_rec, "val", val_dir, val_threads)
        executor.submit(create_rec, "test", test_dir, test_threads)
    print(f"Created .rec files", flush=True)

    # Get identity and image counts
    command_labels = []
    for dataset_path in [train_dir, val_dir, test_dir]:
        for file_type in ["d -not -empty", "l"]:
            command = f"find \"{dataset_path}\" -type {file_type} | wc -l"
            label = f"{dataset_path.split('/')[-2].capitalize()} {'images' if file_type == 'l' else 'identities'}"
            command_labels.append((command, label))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_count, command_labels)

    # Print results
    for label, output in results:
        print(f"{label}: {output}")

if __name__ == "__main__":
    assert len(sys.argv) >= 6
    images_dir = sys.argv[1]
    out_dir = sys.argv[2]
    val_split = float(sys.argv[3])
    test_split = float(sys.argv[4])
    img_uid_map = sys.argv[5]
    max_threads = int(sys.argv[6]) if len(sys.argv) >= 7 else 16
    seed = int(sys.argv[7]) if len(sys.argv) >= 8 else None
    create_dataset(images_dir, out_dir, val_split, test_split, img_uid_map, max_threads, seed)
