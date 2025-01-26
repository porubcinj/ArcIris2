from easydict import EasyDict as edict

config = edict()

# Dataset Paths
config.images_dir = "/Volumes/T7/ArcIris Images"
config.dataset_dir = "/Volumes/T7/ArcIris Image Symlinks"
config.img_uid_map = "/Users/jozef/Documents/personal/Research/ArcIris2/img_to_uid_map.json"

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.save_all_states = False
config.output = "ms1mv3_arcface_r50"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False
config.batch_size = 128

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# For Large Scale Dataset, such as WebFace42M
config.dali = False 
config.dali_aug = False

# Gradient Accumulation
config.gradient_acc = 1

# Setup seed
config.seed = 2048

# Dataload numworkers
config.num_workers = 2