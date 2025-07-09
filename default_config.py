from third_party.yacs_config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.latent_dim = 128
cfg.model.input_dim = 3
cfg.model.n_flows = 4
cfg.model.depth = 21
cfg.model.feat_dim = 64
cfg.model.point_prior_n_layers = 1
cfg.model.weight_n_layers = 1
cfg.model.quantizer = 'kl' # or 'softvq'
cfg.model.ddpm_backbone = 'unet1'  # Options: 'unet1', 'unet1x', 'unet1024'

cfg.model.klquantizer = CN()
cfg.model.klquantizer.kl_weight = 0.5
cfg.model.klquantizer.n_layers = 1

cfg.model.soft_vq = CN()
cfg.model.soft_vq.n_e = 1024  # Number of embeddings
cfg.model.soft_vq.tau = 0.07  # Temperature for softmax
cfg.model.soft_vq.entropy_loss_ratio = 0.01
cfg.model.soft_vq.show_usage = True  # Track codebook usage
cfg.model.soft_vq.l2_norm = False  # Normalize embeddings

cfg.data = CN()
cfg.data.dataset = 'ShapeNetCore.v2'
cfg.data.categories = ['chair']
cfg.data.n_sample_points = 2048
cfg.data.data_dir = './data'  # Add this
cfg.data.batch_size = 32
cfg.data.batch_size_test = 32
cfg.data.num_workers = 4
cfg.data.train_drop_last = 1

cfg.training = CN()
cfg.training.batch_size = 32
cfg.training.epochs = 100
cfg.training.learning_rate = 0.001
cfg.training.type = "trainers.vae_trainer"  # or "trainers.ddpm_trainer"

cfg.training.opt = CN()
cfg.training.opt.type = 'adamw'
cfg.training.opt.lr = 0.001
cfg.training.opt.beta1 = 0.9
cfg.training.opt.beta2 = 0.999
cfg.training.opt.weight_decay = 0.01
cfg.training.opt.ema = True
cfg.training.opt.ema_decay = 0.9999
cfg.training.opt.scheduler = 'cosine_anneal_nocycle'

# Add missing configuration
cfg.exp_name = ''
cfg.log_freq = 100
cfg.viz_freq = 1000
cfg.save_freq = 10
cfg.val_freq = 5
cfg.save_time = 60  # minutes