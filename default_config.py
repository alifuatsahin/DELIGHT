from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.latent_dim = 128
cfg.model.input_dim = 3
cfg.model.n_flows = 4
cfg.model.depth = 21
cfg.model.feat_dim = 64
cfg.model.extra_feature_channels = 0
cfg.model.posterior_n_layers = 1
cfg.model.point_prior_n_layers = 1
cfg.model.weight_n_layers = 3
cfg.model.prior_flow_depth = 7
cfg.model.prior_feat_dim = 128

cfg.data = CN()
cfg.data.dataset = 'shapenet'
cfg.data.n_sample_points = 2048
cfg.data.data_dir = 'path/to/your/data'  # Add this
cfg.data.batch_size = 32
cfg.data.batch_size_test = 32
cfg.data.num_workers = 4
cfg.data.train_drop_last = 1

cfg.training = CN()
cfg.training.batch_size = 32
cfg.training.epochs = 100
cfg.training.learning_rate = 0.001
cfg.training.type = "trainers.vae_trainer"  # or "trainers.ldm_trainer"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.pnll_weight = 1.0
cfg.training.opt.gnll_weight = 1.0
cfg.training.opt.entl_weight = 1.0
cfg.training.opt.ema = True
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = ''

# Add missing configuration
cfg.save_dir = 'outputs'
cfg.log_freq = 100
cfg.viz_freq = 1000
cfg.save_freq = 10
cfg.val_freq = 5
cfg.save_time = 60  # minutes