from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.exp_name = ''

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
cfg.model.high_freq_recon_coeff = 0.0  # Coefficient for high-frequency reconstruction loss
cfg.model.high_freq_recon_lmax = 50  # Lmax for high-frequency reconstruction loss

cfg.model.klquantizer = CN()
cfg.model.klquantizer.n_layers = 1

cfg.model.soft_vq = CN()
cfg.model.soft_vq.n_e = 1024  # Number of embeddings
cfg.model.soft_vq.num_codebooks = 1  # Number of codebooks
cfg.model.soft_vq.learnable = True  # Whether to learn the temperature
cfg.model.soft_vq.tau_min = 0.001  # Minimum temperature
cfg.model.soft_vq.tau_max = 0.1  # Maximum temperature
cfg.model.soft_vq.tau = 0.01  # (Initial) Temperature for softmax
cfg.model.soft_vq.entropy_loss_ratio = 100 # Ratio for entropy loss (0.01)
cfg.model.soft_vq.show_usage = True  # Track codebook usage
cfg.model.soft_vq.l2_norm = False  # Normalize embeddings

cfg.data = CN()
cfg.data.dataset = 'ShapeNetCore.v2'
cfg.data.categories = 'chair'
cfg.data.n_sample_points = 2048
cfg.data.data_dir = './ShapeNet'
cfg.data.batch_size = 32
cfg.data.batch_size_test = 32
cfg.data.num_workers = 10
cfg.data.train_drop_last = 1
cfg.data.random_subsample = True  # Whether to randomly subsample point clouds
cfg.data.normalize_per_shape = False
cfg.data.normalize_shape_box = False
cfg.data.normalize_global = False
cfg.data.normalize_std_per_axis = False
cfg.data.normalize_range = False  # not used
cfg.data.recenter_per_shape = True
cfg.data.sample_with_replacement = True  # Whether to sample points with replacement
cfg.data.dataset_scale = 1.0  # Scale for the dataset

cfg.training = CN()
cfg.training.batch_size = 32
cfg.training.epochs = 1000
cfg.training.anneal_portion = 0.5  # Portion of epochs for annealing
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
cfg.vis = CN()
cfg.vis.log_freq = -1
cfg.vis.vis_freq = -50
cfg.vis.save_freq = 400
cfg.vis.val_freq = 25
cfg.vis.save_time = 30  # minutes