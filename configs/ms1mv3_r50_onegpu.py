from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

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
config.num_classes = 2207
config.num_image = 461090
config.num_epoch = 200
config.warmup_epoch = 0
config.val_targets = []
config.num_workers = 1