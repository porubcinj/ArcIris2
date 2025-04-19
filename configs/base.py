from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.save_all_states = True
config.output = None

config.embedding_size = 512

# Partial FC
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0

config.fp16 = True
config.batch_size = 128

# For SGD
#config.optimizer = "sgd"
#config.lr = 0.001
#config.momentum = 0.9
#config.weight_decay = 5e-4

# For AdamW
config.optimizer = "adamw"
config.lr = 0.001
config.weight_decay = 0.01

config.verbose = 2000 # Number of global steps between logging and running validation
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False 
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# WandB Logger
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False

config.train_dir = "/project01/cvrl/jporubci/ArcIris Dataset/train"
config.val_dir = "/project01/cvrl/jporubci/ArcIris Dataset/val"
config.test_dir = "/project01/cvrl/jporubci/ArcIris Dataset/test"
config.num_classes = 2071
config.num_image = 182644
config.num_epoch = 200
config.warmup_epoch = 0
config.val_target = "val.bin"
config.debug = False
config.strategy = "" # ["", "noise", "blur", "randomly_noise_or_blur"]
