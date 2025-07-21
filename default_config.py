from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.exp_name = ''

cfg.model = CN()
# cfg.model.sa_blocks = [
#         ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
#         ((64, 3, 16), (256, 0.2, 32, (64, 128))),
#         ((128, 3, 8), (128, 0.4, 32, (128, 128))),
#         (None, (64, 0.8, 32, (128, 128, 128))), 
# ]
cfg.model.latent_dim = 1024
cfg.model.input_dim = 3
cfg.model.point_prior_n_layers = 1
cfg.model.num_blocks = 1 # Number of CNF blocks
cfg.model.dims = '512-512-512'  # Hidden dimensions for the CN
cfg.model.layer_type = "concatsquash"
cfg.model.nonlinearity = "tanh"  # Nonlinearity for the CNF
cfg.model.train_T = True
cfg.model.time_length = 0.5  # Time length for the CNF
cfg.model.solver = 'dopri5'  # ODE solver
cfg.model.atol = 1e-5  # Absolute tolerance for the ODE solver
cfg.model.rtol = 1e-5  # Relative tolerance for the ODE solver
cfg.model.use_adjoint = True  # Use adjoint method for ODE solving
cfg.model.batch_norm = True  # Use synchronized batch normalization
cfg.model.bn_lag = 0  # Batch norm lag
cfg.model.sync_bn = False  # Use synchronized batch normalization across GPUs
cfg.model.quantizer = 'kl' # 'kl' or 'softvq'
cfg.model.ddpm_backbone = 'unet1'  # Options: 'unet1', 'unet1x', 'unet1024'
cfg.model.anneal_kl = True  # Whether to use KL annealing
cfg.model.max_kl_coeff = 0.5  # Maximum KL coefficient
cfg.model.min_kl_coeff = 1e-7  # Minimum KL coefficient
cfg.model.constant_portion = 0.0  # Portion of epochs with constant KL coefficient
cfg.model.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.model.kl_weight = 1  # KL weight for quantization loss
cfg.model.high_freq_recon_coeff = 0  # Coefficient for high-frequency reconstruction loss
cfg.model.high_freq_recon_lmax = 50  # Maximum frequency for high-frequency reconstruction

cfg.model.klquantizer = CN()
cfg.model.klquantizer.n_layers = 1

cfg.model.soft_vq = CN()
cfg.model.soft_vq.n_e = 64  # Number of embeddings
cfg.model.soft_vq.e_dim = 16  # Embedding dimension
cfg.model.soft_vq.num_codebooks = 64  # Number of codebooks
cfg.model.soft_vq.learnable = True  # Whether to learn the temperature
cfg.model.soft_vq.tau_min = 0.01  # Minimum temperature
cfg.model.soft_vq.tau_max = 0.2  # Maximum temperature
cfg.model.soft_vq.tau = 0.07  # (Initial) Temperature for softmax
cfg.model.soft_vq.entropy_loss_ratio = 100 # Ratio for entropy loss (0.01)
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
cfg.training.type = "vae"  # or "ddpm"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.ema = False
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = 'cosine_anneal'

cfg.vis = CN()
cfg.vis.log_freq = -1
cfg.vis.vis_freq = -50
cfg.vis.save_freq = 400
cfg.vis.val_freq = 25
cfg.vis.save_time = 30  # minutes