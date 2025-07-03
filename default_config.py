from third_party.yacs_config import CfgNode as CN

cfg = CN()
cfg.model.latent_dim = 256
cfg.model.input_dim = 3
cfg.model.n_flows = 4
cfg.model.depth = 64
cfg.model.feat_dim = 63
cfg.model.extra_feature_channels = 0
cfg.model.posterior_n_layers = 1
cfg.model.point_prior_n_layers = 3
cfg.model.weight_n_layers = 2
cfg.training.batch_size = 32
cfg.training.epochs = 100
cfg.training.learning_rate = 0.001