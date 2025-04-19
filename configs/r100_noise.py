from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.save_all_states = True
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.optimizer = "adamw"
config.weight_decay = 0.01
config.batch_size = 128
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.train_dir = "/project01/cvrl/jporubci/ArcIris Dataset/train"
config.val_dir = "/project01/cvrl/jporubci/ArcIris Dataset/val"
config.test_dir = "/project01/cvrl/jporubci/ArcIris Dataset/test"
config.num_classes = 2071
config.num_image = 182644
config.num_epoch = 200
config.warmup_epoch = 0
config.val_target = "val.bin"
config.num_workers = 2
config.debug = True
config.strategy = "noise" # ["", "noise", "blur", "randomly_noise_or_blur"]
