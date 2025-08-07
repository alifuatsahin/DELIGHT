from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.exp_name = ''
cfg.save_dir = ''  # Directory to save logs and models

cfg.vae = CN()
cfg.vae.latent_dim = 512
cfg.vae.input_dim = 3
cfg.vae.n_flows = 4
cfg.vae.depth = 21
cfg.vae.feat_dim = 64
cfg.vae.point_prior_n_layers = 1
cfg.vae.weight_n_layers = 3
cfg.vae.quantizer = 'softvq' # or 'softvq'
cfg.vae.ddpm_backbone = 'unet1'  # Options: 'unet1', 'unet1x', 'unet1024'
cfg.vae.anneal_kl = False  # Whether to use KL annealing
cfg.vae.max_kl_coeff = 0.5  # Maximum KL coefficient
cfg.vae.min_kl_coeff = 1e-7  # Minimum KL coefficient
cfg.vae.constant_portion = 0.0  # Portion of epochs with constant KL coefficient
cfg.vae.anneal_portion = 0.5  # Portion of epochs for annealing
cfg.vae.kl_weight = 1  # KL weight for quantization loss
cfg.vae.high_freq_recon_coeff = 0.0  # Coefficient for high-frequency reconstruction loss
cfg.vae.high_freq_recon_lmax = 50  # Lmax for high-frequency reconstruction loss

cfg.vae.soft_vq = CN()
cfg.vae.soft_vq.n_e = 64  # Number of embeddings per codebook
cfg.vae.soft_vq.e_dim = 32  # Dimension of each embedding
cfg.vae.soft_vq.num_codebooks = 32  # Number of codebooks
cfg.vae.soft_vq.learnable = True  # Whether to learn the temperature
cfg.vae.soft_vq.tau_min = 0.03  # Minimum temperature
cfg.vae.soft_vq.tau_max = 0.3  # Maximum temperature
cfg.vae.soft_vq.tau = 0.07  # (Initial) Temperature for softmax
cfg.vae.soft_vq.entropy_loss_ratio = 100 # Ratio for entropy loss (0.01)
cfg.vae.soft_vq.show_usage = True  # Track codebook usage
cfg.vae.soft_vq.l2_norm = True  # Normalize embeddings

cfg.ddpm = CN()
cfg.ddpm.model_channels = 320
cfg.ddpm.num_res_blocks = 3
cfg.ddpm.attention_resolutions = (2, 4, 8)
cfg.ddpm.dropout = 0.0
cfg.ddpm.channel_mult = (1, 2, 4, 4)
cfg.ddpm.conv_resample = True
cfg.ddpm.dims = 1
cfg.ddpm.num_classes = -1  # No class conditioning
cfg.ddpm.use_checkpoint = False
cfg.ddpm.use_fp16 = False
cfg.ddpm.num_heads = 4
cfg.ddpm.num_head_channels = -1
cfg.ddpm.num_heads_upsample = -1
cfg.ddpm.use_scale_shift_norm = False
cfg.ddpm.resblock_updown = False
cfg.ddpm.use_xformers_attention = True

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
cfg.training.batch_size = 32
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