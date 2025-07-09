from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.latent_dim = 512
cfg.model.input_dim = 3
cfg.model.n_flows = 4
cfg.model.depth = 21
cfg.model.feat_dim = 64
cfg.model.point_prior_n_layers = 1
cfg.model.weight_n_layers = 1
cfg.model.quantizer = 'softvq' # or 'softvq'
cfg.model.ddpm_backbone = 'unet1'  # Options: 'unet1', 'unet1x', 'unet1024'
cfg.model.anneal_kl = True  # Whether to use KL annealing
cfg.model.max_kl_coeff = 0.5  # Maximum KL coefficient
cfg.model.min_kl_coeff = 1e-7  # Minimum KL coefficient
cfg.model.constant_portion = 0.0  # Portion of epochs with constant KL coefficient
cfg.model.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.model.kl_weight = 1  # KL weight for quantization loss

cfg.model.klquantizer = CN()
cfg.model.klquantizer.n_layers = 1

cfg.model.soft_vq = CN()
cfg.model.soft_vq.n_e = 1024  # Number of embeddings
cfg.model.soft_vq.learnable = True  # Whether to learn the temperature
cfg.model.soft_vq.tau_min = 0.01  # Minimum temperature
cfg.model.soft_vq.tau_max = 1.0  # Maximum temperature
cfg.model.soft_vq.tau = 0.1  # (Initial) Temperature for softmax
cfg.model.soft_vq.entropy_loss_ratio = 1 # Ratio for entropy loss (0.01)
cfg.model.soft_vq.show_usage = True  # Track codebook usage
cfg.model.soft_vq.l2_norm = False  # Normalize embeddings

cfg.data = CN()
cfg.data.dataset = 'ShapeNetCore.v2'
cfg.data.categories = 'chair'
cfg.data.n_sample_points = 2048
cfg.data.data_dir = './data'
cfg.data.batch_size = 32
cfg.data.batch_size_test = 32
cfg.data.num_workers = 10
cfg.data.train_drop_last = 1

cfg.training = CN()
cfg.training.batch_size = 32
cfg.training.epochs = 1000
cfg.training.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.training.learning_rate = 0.001
cfg.training.warmup = 10 # warmup epochs for mixture weights
cfg.training.type = "vae"  # or "ddpm"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.ema = False
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = 'cosine_anneal_nocycle'

# Add missing configuration
cfg.exp_name = ''
cfg.log_freq = 50
cfg.viz_freq = 100
cfg.save_freq = 50
cfg.val_freq = 50
cfg.save_time = 30  # minutes