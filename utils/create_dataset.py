import json
import os
import shutil
import tempfile

def create_dataset(images_dir, dataset_dir, img_uid_map):
    if os.path.exists(dataset_dir):
        return

    with open(img_uid_map, 'r') as file:
        json_data = json.load(file)

    dataset_map = {}
    for path, identity in json_data.items():
        root, ext = os.path.splitext(path)
        paths = f'{root}_p{ext}', f'{root}_pm{ext}'
        if all(os.path.exists(os.path.join(images_dir, path)) for path in paths):
            if identity not in dataset_map:
                dataset_map[identity] = set()
            dataset_map[identity].add(paths)

    with tempfile.TemporaryDirectory() as temp_dir:
        for identity in dataset_map.keys():
            os.makedirs(os.path.join(temp_dir, identity))

        for identity, paths in dataset_map.items():
            for i, (p_path, pm_path) in enumerate(paths):
                os.symlink(p_path, os.path.join(temp_dir, identity, f'{i}.png'))

        shutil.move(temp_dir, dataset_dir)