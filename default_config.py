from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.exp_name = ''
cfg.save_dir = ''  # Directory to save logs and models

cfg.vae = CN()
cfg.vae.latent_dim = 512
cfg.vae.input_dim = 3
cfg.vae.point_prior_n_layers = 1
cfg.vae.weight_n_layers = 3
cfg.vae.quantizer = 'softvq' # 'kl' or 'softvq'
cfg.vae.anneal_kl = False  # Whether to use KL annealing
cfg.vae.max_kl_coeff = 0.5  # Maximum KL coefficient
cfg.vae.min_kl_coeff = 1e-7  # Minimum KL coefficient
cfg.vae.constant_portion = 0.0  # Portion of epochs with constant KL coefficient
cfg.vae.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.vae.kl_weight = 1  # KL weight for quantization loss

cfg.vae.flow = CN()
cfg.vae.flow.depth = 3  # Depth of the flow model
cfg.vae.flow.t_emb_ch = 3 # Number of channels for time embedding
cfg.vae.flow.n_flows = 2  # Number of flow priors
cfg.vae.flow.width = 512  # Feature dimension for the flow model
cfg.vae.flow.use_hybrid_coupling = True  # Whether to use hybrid coupling
cfg.vae.flow.beta = 0.2  # Beta for the hybrid coupling
cfg.vae.flow.n_heads = 8 # Number of attention heads
cfg.vae.flow.num_res_blocks = 1  # Number of residual blocks in the flow
cfg.vae.flow.attn_depth = 1  # Depth of attention layers
cfg.vae.flow.dim_head = 64  # Dimension of each attention head
cfg.vae.flow.cfm_method = 'exact_ot'  # Method for conditional flow matching
cfg.vae.flow.solver = 'dopri5'  # ODE solver ('dopri5', 'rk4', etc.)
cfg.vae.flow.atol = 1e-4  # Absolute tolerance for ODE
cfg.vae.flow.rtol = 1e-4  # Relative tolerance for ODE
cfg.vae.flow.sigma = 0.01  # Sigma for the flow
cfg.vae.flow.use_xformers_attention = True  # Whether to use xformers for attention
cfg.vae.flow.ot_method = 'exact'  # Method for optimal transport ('sinkhorn', 'exact', 'partial', 'unbalanced')

cfg.vae.softvq = CN()
cfg.vae.softvq.n_e = 64  # Number of embeddings per codebook
cfg.vae.softvq.e_dim = 32  # Dimension of each embedding
cfg.vae.softvq.num_codebooks = 32  # Number of codebooks
cfg.vae.softvq.learnable = True  # Whether to learn the temperature
cfg.vae.softvq.tau_min = 0.03  # Minimum temperature
cfg.vae.softvq.tau_max = 0.3  # Maximum temperature
cfg.vae.softvq.tau = 0.07  # (Initial) Temperature for softmax
cfg.vae.softvq.entropy_loss_ratio = 0.01 # Ratio for entropy loss (0.01)
cfg.vae.softvq.show_usage = True  # Track codebook usage
cfg.vae.softvq.l2_norm = True  # Normalize embeddings

cfg.ddpm = CN()
cfg.ddpm.model_channels = 256
cfg.ddpm.num_res_blocks = 3
cfg.ddpm.attention_resolutions = (2, 4, 8)
cfg.ddpm.dropout = 0.0
cfg.ddpm.channel_mult = (1, 2, 4, 4)
cfg.ddpm.conv_resample = True
cfg.ddpm.dims = 1
cfg.ddpm.num_classes = -1  # No class conditioning
cfg.ddpm.use_checkpoint = False
cfg.ddpm.num_heads = 4
cfg.ddpm.num_head_channels = -1
cfg.ddpm.num_heads_upsample = -1
cfg.ddpm.use_scale_shift_norm = False
cfg.ddpm.resblock_updown = False
cfg.ddpm.use_xformers_attention = True
cfg.ddpm.parameterization = 'eps'  # 'eps' or 'x0'
cfg.ddpm.original_elbo_weight = 0.0  # Weight for original ELBO loss
cfg.ddpm.l_simple_weight = 1.0  # Weight for simple loss
cfg.ddpm.learn_logvar = False  # Whether to learn log variance
cfg.ddpm.logvar_init = 0.0  # Initial value for log variance
cfg.ddpm.loss_type = 'l2'  # Loss type for training ('l1', 'l2')
cfg.ddpm.beta_schedule = 'linear'  # Beta schedule for diffusion process ('linear', 'cosine', 'sqrt_linear', 'sqrt')
cfg.ddpm.timesteps = 1000  # Number of diffusion timesteps
cfg.ddpm.v_posterior = 0.0  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
cfg.ddpm.clip_denoised = True  # Whether to clip denoised samples
cfg.ddpm.log_every_t = 100  # Frequency of logging during sampling

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
cfg.data.normalize_range = False  # not used::
cfg.data.recenter_per_shape = True
cfg.data.sample_with_replacement = True  # Whether to sample points with replacement
cfg.data.dataset_scale = 1.0  # Scale for the dataset

cfg.training = CN()
cfg.training.epochs = 500
cfg.training.warmup = 10 # warmup epochs for mixture weights
cfg.training.type = "vae"  # or "ddpm"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.0001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.ema = True
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = 'step'  # 'cosine_anneal', 'exponential', 'step', 'linear', 'lambda', 'cosine_anneal_nocycle'

# Add missing configuration
cfg.vis = CN()
cfg.vis.log_freq = -1
cfg.vis.vis_freq = -50
cfg.vis.save_freq = 400
cfg.vis.val_freq = 25
cfg.vis.save_time = 30  # minutes