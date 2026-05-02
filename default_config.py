from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.exp_name = ''
cfg.save_dir = ''  # Directory to save logs and models

cfg.vae = CN()
cfg.vae.latent_dim = 512
cfg.vae.input_dim = 3
cfg.vae.quantizer = 'softvq' # 'kl' or 'softvq'
cfg.vae.anneal_kl = False  # Whether to use KL annealing
cfg.vae.max_kl_coeff = 1.0  # Maximum KL coefficient
cfg.vae.min_kl_coeff = 1e-7  # Minimum KL coefficient
cfg.vae.constant_portion = 0.0  # Portion of epochs with constant KL coefficient
cfg.vae.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.vae.kl_weight = 1.0  # KL weight for quantization loss

cfg.vae.flow = CN()
cfg.vae.flow.depth = 3  # Depth of the flow model
cfg.vae.flow.widths = (48, 96, 192, 384)  # Feature dimensions for each layer
cfg.vae.flow.n_heads = (2, 4, 8, 16)  # Number of attention heads
cfg.vae.flow.patch_size = 32  # Patch size for local attention
cfg.vae.flow.num_centers = (1024, 256, 128)  # Number of centers for each layer
cfg.vae.flow.radiuses = (0.1, 0.2, 0.4)  # Radii for each layer
cfg.vae.flow.num_neighbors = (32, 32, 32)  # Number of neighbors for each layer
cfg.vae.flow.t_emb_ch = 4 # Number of channels for time embedding
cfg.vae.flow.p = 2  # p-norm for the flow
cfg.vae.flow.blur = 0.05  # Gaussian blur for skinhorn regularization
cfg.vae.flow.use_hybrid_coupling = True  # Whether to use hybrid coupling
cfg.vae.flow.beta = 0.2  # Beta for the hybrid coupling
cfg.vae.flow.cfm_method = 'variance'  # Method for conditional flow matching
cfg.vae.flow.solver = 'dopri5'  # ODE solver ('dopri5', 'rk4', etc.)
cfg.vae.flow.atol = 1e-4  # Absolute tolerance for ODE
cfg.vae.flow.rtol = 1e-4  # Relative tolerance for ODE
cfg.vae.flow.sigma = 0.0  # Sigma for the flow
cfg.vae.flow.use_xformers = True  # Whether to use xformers for attention

cfg.vae.softvq = CN()
cfg.vae.softvq.n_e = 64  # Number of embeddings per codebook
cfg.vae.softvq.e_dim = 32  # Dimension of each embedding
cfg.vae.softvq.num_codebooks = 32  # Number of codebooks
cfg.vae.softvq.learnable = True  # Whether to learn the temperature
cfg.vae.softvq.tau = 0.07  # (Initial) Temperature for softmax
cfg.vae.softvq.entropy_loss_ratio = 0.01 # Ratio for entropy loss (0.01)
cfg.vae.softvq.show_usage = True  # Track codebook usage
cfg.vae.softvq.l2_norm = True  # Normalize embeddings

cfg.prior = CN()
cfg.prior.cfm_method = 'variance'  # Type of prior ('ot', 'schrodinger_bridge', etc.)
cfg.prior.depth = 3  # Number of layers in prior
cfg.prior.width = 256  # Feature dimension for the flow model
cfg.prior.t_emb_ch = 6 # Number of channels for time embedding
cfg.prior.use_hybrid_coupling = True  # Whether to use hybrid coupling
cfg.prior.beta = 0.2  # Beta for the hybrid coupling
cfg.prior.solver = 'dopri5'  # ODE solver ('dopri5', 'rk4', etc.)
cfg.prior.atol = 1e-4  # Absolute tolerance for ODE
cfg.prior.rtol = 1e-4  # Relative tolerance for ODE
cfg.prior.sigma = 0.0  # Sigma for the flow
cfg.prior.p = 2  # p-norm
cfg.prior.blur = 0.05  # Gaussian blur for skinhorn regularization
cfg.prior.dim_head = 256  # Dimension per attention head
cfg.prior.n_heads = 4  # Number of attention heads
cfg.prior.use_xformers = True  # Whether to use xformers for attention

cfg.data = CN()
cfg.data.dataset = 'ShapeNetCore.v2'
cfg.data.categories = 'chair'
cfg.data.n_sample_points = 2048
cfg.data.superset_size = 10000
cfg.data.batch_size = 32
cfg.data.batch_size_test = 32
cfg.data.num_workers = 10
cfg.data.train_drop_last = 1
cfg.data.random_subsample = True  # Whether to randomly subsample point clouds
cfg.data.normalize_per_shape = False
cfg.data.normalize_shape_box = False
cfg.data.normalize_global = True  # Normalize point clouds globally
cfg.data.normalize_std_per_axis = False
cfg.data.recenter_per_shape = False
cfg.data.sample_with_replacement = True  # Whether to sample points with replacement
cfg.data.dataset_scale = 1.0  # Scale for the dataset

cfg.training = CN()
cfg.training.epochs = 500
cfg.training.type = "vae"  # or "prior"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.0001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.ema = True
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = 'cosine_anneal_nocycle'  # 'cosine_anneal', 'exponential', 'step', 'linear', 'lambda', 'cosine_anneal_nocycle'

# Add missing configuration
cfg.vis = CN()
cfg.vis.log_freq = -1
cfg.vis.vis_freq = -50
cfg.vis.save_freq = 50
cfg.vis.val_freq = 25
cfg.vis.save_time = 30  # minutes