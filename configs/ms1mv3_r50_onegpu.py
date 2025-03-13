from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.optimizer = "adamw"
config.momentum = 0.9
config.weight_decay = 0.01
config.batch_size = 128
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.rec = "."
config.num_classes = 1657
config.num_image = 161335
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = []
config.image_size = (64, 512)
config.num_workers = 1