import os
import time
import torch
import torchvision
from abc import ABC, abstractmethod
from contextlib import contextmanager
import numpy as np
from loguru import logger
import torch.distributed as dist

from datasets import get_data_loaders
from utils.utils import AverageMeter
from utils.vis_helper import visualize_point_clouds_3d
from utils.data_helper import normalize_point_clouds
from utils.eval_helper import compute_NLL_metric, get_ref_num, compute_score

class BaseTrainer(ABC):
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else 'cpu'
        self.device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.train_loader, self.test_loader = None, None
        self.local_rank = args.local_rank
        self.writer = None
        self.ema = None
        self.num_points = cfg.data.n_sample_points
        self.use_ema = cfg.training.opt.ema
        self.best_eval_epoch = 0
        self.best_eval_score = -1
        self.start_epoch = 1

    @abstractmethod
    def train_iter(self, batch, step):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def log_loss(self, loss_dict, writer=None, step=None):
        """Log loss values to writer"""
        if writer is not None:
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value, step)

    def filter_name(self, ckpt):
        ckpt_new = {}
        for k, v in ckpt.items():
            if k[:7] == 'module.':
                kn = k[7:]
            elif k[:13] == 'model.module.':
                kn = k[13:]
            elif k[:6] == 'module':
                kn = k[6:]
            else:
                kn = k
            ckpt_new[kn] = v
        return ckpt_new

    def add_module_prefix(self, state_dict):
        """Add 'module.' prefix to every key in a state dict."""
        state_dict_new = {}
        for k, v in state_dict.items():
            # Avoid double prefixing if already present
            if not k.startswith('module.'):
                kn = 'module.' + k
            else:
                kn = k
            state_dict_new[kn] = v
        return state_dict_new

    def set_writer(self, writer):
        self.writer = writer

    def epoch_end(self, epoch, writer=None):
        # Signal now that the epoch ends....
        if self.scheduler is not None:
            self.scheduler.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.scheduler.get_last_lr()[0], epoch)
        if writer is not None:
            writer.upload_meter(step=epoch)

    def build_data(self):
        logger.info('Building data loader...')
        if self.cfg.training.type == 'vae' and (self.cfg.vae.flow.cfm_method == 'ot' or self.cfg.vae.flow.cfm_method == 'schrodinger_bridge'):
            return_superset = True
        else:
            return_superset = False
        loaders = get_data_loaders(self.cfg.data, self.args, return_superset=return_superset)
        train_loader = loaders['train_loader']
        test_loader = loaders['test_loader']

        return train_loader, test_loader

    def train_epochs(self):
        cfg, args = self.cfg, self.args
        writer = self.writer
        train_loader = self.train_loader

        if cfg.vis.log_freq <= -1:  # treat as per epoch
            cfg.vis.log_freq = int(- cfg.vis.log_freq * len(train_loader))
        if cfg.vis.vis_freq <= -1:
            cfg.vis.vis_freq = - cfg.vis.vis_freq * len(train_loader)

        logger.info('[GPU {}] Starting training for {} epochs'.format(args.local_rank, cfg.training.epochs))

        tic_global = time.time()
        if args.global_rank == 0:
            tic_log = time.time()
            start_time = time.time()
        avg_time = AverageMeter()
        step = 0

        self.total_iter = cfg.training.epochs * len(train_loader)
        if hasattr(self.model, 'module'):
            self.model.module.total_iter = self.total_iter
        else:
            # For single GPU models
            self.model.total_iter = self.total_iter

        for epoch in range(self.start_epoch, cfg.training.epochs + 1):
            self.model.train()

            if args.global_rank == 0:
                tic_epo = time.time()
            epoch_loss = []
           
            for idx, batch in enumerate(train_loader):
                step = idx + len(train_loader) * epoch

                if args.global_rank == 0 and self.writer is not None:
                    tic_iter = time.time()

                loss = self.train_iter(batch, step=step)

                if args.global_rank == 0:
                    epoch_loss.append(loss)

                if self.args.global_rank == 0 and (
                        time.time() - tic_log > 60
                ):  # log per min
                    logger.info(
                        f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.4f} | '
                        f'[exp] {cfg.save_dir} | [step] {step:5d}'
                    )
                    tic_log = time.time()

                # -- visualize rec and samples -- #
                if step % int(cfg.vis.log_freq) == 0 and args.global_rank == 0 and not step == 0:
                    avg_loss = np.array(epoch_loss).mean()
                    epoch_loss = []  # clean up epoch loss
                    self.log_loss({'epo_loss': avg_loss},
                                  writer=writer, step=step)
                    visualize = int(cfg.vis.vis_freq) > 0 and \
                        (step) % int(cfg.vis.vis_freq) == 0
                    if visualize:
                        self.vis_recont(batch, writer, step)
                        self.model.eval()
                        self.vis_sample(writer, step)
                        self.model.train()

                # -- timer -- #
                if args.global_rank == 0 and self.writer is not None:
                    time_iter = time.time() - tic_iter
                    self.writer.avg_meter('time_iter', time_iter, step=step)

            if args.global_rank == 0 and self.writer is not None:
                epo_time = (time.time() - tic_epo) / 60.0  # min
                avg_time.update(epo_time)
                logger.info(
                    f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.4f} | '
                    f'[exp] {cfg.save_dir} | [step] {step:5d} | [time] {epo_time:.1f}m (~{int(avg_time.avg * (cfg.training.epochs - epoch) / 60)}h) | '
                    f'[best] {self.best_eval_epoch} {self.best_eval_score * 1e2:.3f}x1e-2'
                )
                tic_log = time.time()

            if epoch % int(cfg.vis.save_freq) == 0 and int(cfg.vis.save_freq) > 0 and args.global_rank == 0:
                save_path = self.save(epoch=epoch, step=step)
                logger.info(f"Checkpoint saved at {save_path} [Epoch] {epoch}")
            
            if (time.time() - tic_global) / 60 > cfg.vis.save_time and args.global_rank == 0:
                save_path = self.save(epoch=epoch, step=step, save_name='snapshot.pth')
                logger.info(f"Checkpoint saved at {save_path}, [Time] {(time.time() - start_time) / 60}h")
                tic_global = time.time()

            if int(cfg.vis.val_freq) > 0 and epoch % int(cfg.vis.val_freq) == 0 and args.global_rank == 0:
                score = self.eval_nll(step=epoch)
                if score < self.best_eval_score or self.best_eval_score < 0:
                    self.save(save_name='best_eval.pth',
                              epoch=epoch, step=step)
                    self.best_eval_score = score
                    self.best_eval_epoch = epoch

            self.epoch_end(epoch, writer=writer)

        if args.global_rank == 0:
            logger.info(f'Training finished, total time: {(time.time() - start_time) / 60:.2f} min')
            logger.info(f'Best eval score: {self.best_eval_score * 1e2:.3f}x1e-2 at epoch {self.best_eval_epoch}')
            self.writer.close() if hasattr(self, 'writer') else None

        if self.cfg.training.type == 'prior':
            logger.info('Starting evaluation of generation...')
            self.model.eval()
            self.eval_sample(step)
            logger.info('Evaluation of generation completed.')

    @torch.no_grad()
    def vis_recont(self, batch, writer=None, step=None):
        """ Visualize reconstruction results """
        
        input = batch['tr_points'].to(self.device)
        output, labels = self.eval(input)

        assert len(input.shape) == len(output.shape) == 3 # (B, Npoints, 3)
        assert input.shape[0] == output.shape[0]  # batch size should match

        nvis = min(max(input.shape[0], 2), 5) # visualize at most 5, at least 2 samples

        img_list = []
        for b in range(nvis):
            x_list, name_list, label_list = [], [], []
            x_list.append(output[b])
            name_list.append('Reconstruction')

            x_list.append(input[b])
            name_list.append('Ground Truth')

            label_list.append(labels[b])
            label_list.append(None) # No label for ground truth

            x_list = normalize_point_clouds(x_list)

            img = visualize_point_clouds_3d(x_list, name_list, labels=label_list)

            img_list.append(img)

        img_list = torchvision.utils.make_grid(
            [torch.as_tensor(a) for a in img_list], pad_value=0)
        
        writer.add_image('vis_out/recont-train', img_list, step)

    @torch.no_grad()
    def vis_sample(self, writer=None, step=None):
        """ Visualize sampling results """
        n_sampled_points = self.num_points
        n_samples = 10

        samples, labels = self.sample(n_sampled_points, n_samples)

        img_list = []
        for idx, (sample, label) in enumerate(zip(samples, labels)):
            x_list, name_list, label_list = [], [], []

            x_list.append(sample)
            name_list.append(f"Sample {idx + 1}")

            label_list.append(label)

            x_list = normalize_point_clouds(x_list)

            img = visualize_point_clouds_3d(x_list, name_list, labels=label_list)

            img_list.append(img)

        img_list = torchvision.utils.make_grid(
            [torch.as_tensor(a) for a in img_list], pad_value=0)
        
        writer.add_image('vis_out/samples', img_list, step)


    # -- shared method for all model with vae component -- #
    @torch.no_grad()
    def eval_nll(self, step=None):
        device = self.device

        gen_pcs, ref_pcs, label_pcs = [], [], []

        data_loader = self.test_loader  # Use validation loader for evaluation

        for vid, val_batch in enumerate(data_loader):
            if vid % 30 == 1:
                logger.info('eval: {}/{}', vid, len(data_loader))

            val_x = val_batch['tr_points'].to(device)
            
            # Check if normalization data exists in batch
            m = val_batch['mean']
            s = val_batch['std']

            B, N, C = val_x.shape
            m = m.view(B, 1, -1)
            s = s.view(B, 1, -1)

            gen_x, labels = self.eval(val_x)

            gen_x = gen_x.cpu()
            val_x = val_x.cpu()
            gen_x[:, :, :3] = gen_x[:, :, :3] * s + m
            val_x[:, :, :3] = val_x[:, :, :3] * s + m
            gen_pcs.append(gen_x.detach().cpu())
            ref_pcs.append(val_x.detach().cpu())
            label_pcs.append(labels.detach().cpu())

        gen_pcs = torch.cat(gen_pcs, dim=0)
        ref_pcs = torch.cat(ref_pcs, dim=0)
        label_pcs = torch.cat(label_pcs, dim=0)

        # Save
        if self.writer is not None:
            img_list = []
            for i in range(10):
                points = gen_pcs[i]
                points = normalize_point_clouds([points])[0]
                label_points = label_pcs[i]
                img = visualize_point_clouds_3d([points], bound=1.0, labels=[label_points])
                img_list.append(img)
            img = np.concatenate(img_list, axis=2)
            self.writer.add_image('nll/rec', torch.as_tensor(img), step)

        results = compute_NLL_metric(
            gen_pcs[:, :, :3], ref_pcs[:, :, :3], label_pcs, device, self.writer, batch_size=20, step=step)
        score = 0
        
        for n, v in results.items():
            if self.writer is not None:
                logger.info('add: {}', n)
                self.writer.add_scalar(f"eval/{n}", v, step)
            if 'CD' in n:
                score = v

        return score
    
    @torch.no_grad()
    def eval_sample(self, step=0):
        """ compute sample metric: MMD, COV, 1-NNA """
        writer = self.writer
        batch_size_test = self.cfg.data.batch_size_test
        device = self.device
        test_loader = self.test_loader
        sample_num_points = self.cfg.data.n_sample_points
        cates = getattr(self.cfg.data, 'categories', ['general'])
        
        # Use get_ref_num to determine number of samples based on category
        if len(cates) == 1 and cates[0] in ['airplane', 'chair', 'car', 'animal', 'mug', 'bottle', 'all']:
            num_samples = get_ref_num(cates[0])
            logger.info(f'Using standard dataset size for {cates[0]}: {num_samples} samples')
        else:
            # Fallback for custom datasets or multiple categories
            num_samples = getattr(self.cfg, 'num_eval_samples', 50)
            logger.info(f'Using configured number of samples: {num_samples}')
        
        logger.info(f'Starting sample evaluation with {num_samples} samples')
        
        # Create category-specific output directories
        for category in cates:
            category_dir = os.path.join(self.cfg.save_dir, 'eval_samples', category)
            os.makedirs(category_dir, exist_ok=True)
        
        # Generate samples
        gen_pcs = []
        ref_pcs = []
        labels_list = []

        # Calculate number of batches needed
        len_test_loader = num_samples // batch_size_test + 1
        
        if self.args.distributed:
            num_gen_iter = max(1, len_test_loader // self.args.global_size)
            if num_gen_iter * batch_size_test * self.args.global_size < num_samples:
                num_gen_iter = num_gen_iter + 1
        else:
            num_gen_iter = len_test_loader
        
        logger.info(f'Rank={self.args.global_rank}, num_gen_iter: {num_gen_iter}; num_samples={num_samples}, batch_size_test={batch_size_test}')
        
        # Generate samples
        seed = getattr(self.cfg, 'eval_seed', 42)
        for i in range(num_gen_iter):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            torch.cuda.manual_seed_all(seed + i)
            
            logger.info(f'Generating batch {i+1}/{num_gen_iter}')
            
            # Generate samples using your model's sample method
            samples, labels = self.sample(sample_num_points, batch_size_test)
            
            gen_pcs.append(samples.detach().cpu())
            labels_list.append(labels.detach().cpu())
        
        # Collect reference data from test loader
        ref_mean_pcs, ref_std_pcs = [], []
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= len_test_loader:
                break
            ref_data = batch['tr_points'].cpu()
            ref_pcs.append(ref_data)
            
            # Collect normalization parameters if available
            m = batch['mean']
            s = batch['std']
            ref_mean_pcs.append(m)
            ref_std_pcs.append(s)

        gen_pcs = torch.cat(gen_pcs, dim=0)
        labels_list = torch.cat(labels_list, dim=0)

        # Handle distributed training
        if self.args.distributed:
            gen_pcs = gen_pcs.to(device)
            labels_list = labels_list.to(device)
            logger.info(f'Before gather: {gen_pcs.shape}, rank={self.args.global_rank}')
            
            gen_pcs_list = [torch.zeros_like(gen_pcs) for _ in range(self.args.global_size)]
            labels_list_gathered = [torch.zeros_like(labels_list) for _ in range(self.args.global_size)]
            dist.all_gather(gen_pcs_list, gen_pcs)
            dist.all_gather(labels_list_gathered, labels_list)
            gen_pcs = torch.cat(gen_pcs_list, dim=0).cpu()
            labels_list = torch.cat(labels_list_gathered, dim=0).cpu()
            
            logger.info(f'After gather: {gen_pcs.shape}, rank={self.args.global_rank}')
        
        # Concatenate all samples and normalization data
        gen_pcs = gen_pcs[:num_samples]
        ref_pcs = torch.cat(ref_pcs, dim=0)[:num_samples]
        labels_list = labels_list[:num_samples]
        ref_mean_pcs = torch.cat(ref_mean_pcs, dim=0)[:num_samples]
        ref_std_pcs = torch.cat(ref_std_pcs, dim=0)[:num_samples]
        
        # Only rank 0 does evaluation and saves results
        if self.args.global_rank != 0:
            return
        
        # Apply denormalization like compute_score does
        if ref_mean_pcs is not None and ref_std_pcs is not None:
            logger.info('Applying denormalization to point clouds')
            # Denormalize to original scale/position
            ref_pcs = ref_pcs * ref_std_pcs + ref_mean_pcs
            gen_pcs = gen_pcs * ref_std_pcs + ref_mean_pcs
        
        # Option for post-processing normalization (like compute_score)
        norm_box = getattr(self.cfg.data, 'normalize_for_eval', False)
        if norm_box:
            logger.info('Applying box normalization for fair comparison')
            ref_pcs = 0.5 * torch.stack(normalize_point_clouds(ref_pcs), dim=0)
            gen_pcs = 0.5 * torch.stack(normalize_point_clouds(gen_pcs), dim=0)
        
        logger.info(f'Final data shapes - ref_pcs: {ref_pcs.shape}, gen_pcs: {gen_pcs.shape}')
        
        # Save generated samples by category
        for category in cates:
            category_dir = os.path.join(self.cfg.save_dir, 'eval_samples', category)
            output_name = os.path.join(category_dir, f'samples_step_{step}.pt')
            torch.save(gen_pcs, output_name)
            logger.info(f'Saved samples for category {category} at {output_name}')
        
        # Visualize samples
        if writer is not None:
            img_list = []
            vis_samples = gen_pcs[:8]  # Visualize first 8 samples
            norm_samples = normalize_point_clouds([s for s in vis_samples])
            img = visualize_point_clouds_3d(norm_samples, [f'sample-{i}' for i in range(len(norm_samples))], labels=labels_list[:8])
            img_list.append(torch.as_tensor(img) / 255.0)
            
            grid = torchvision.utils.make_grid(img_list)
            writer.add_image('eval_samples/generated', grid, step)
                
        results = {}
        for category in cates:
            logger.info(f'Computing metrics for category: {category}')
            
            # Compute comprehensive metrics like compute_score does
            category_results = compute_score(
                gen_pcs[:, :, :3].to(device).float(), 
                ref_pcs[:, :, :3].to(device).float(), 
                batch_size_test=min(50, batch_size_test),
                accelerated_cd=True, device_str=self.device_str,
                visualize=True, writer=writer,
            )
            
            results[category] = category_results
            
            # Save metrics to category folder
            category_dir = os.path.join(self.cfg.save_dir, 'eval_samples', category)
            metrics_file = os.path.join(category_dir, f'metrics_step_{step}.txt')
            
            with open(metrics_file, 'w') as f:
                f.write(f'Evaluation Results for {category} at step {step}\n')
                f.write('=' * 50 + '\n')
                for metric_name, value in category_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f'{metric_name}: {value:.6f}\n')
                    else:
                        f.write(f'{metric_name}: {value}\n')
                    
                    # Log to tensorboard
                    if writer is not None and isinstance(value, (int, float)):
                        writer.add_scalar(f'eval/{category}/{metric_name}', value, step)
            
            logger.info(f'Saved metrics for {category} at {metrics_file}')
        
        # Log summary
        msg = f'Evaluation completed at step {step}\n'
        for category, category_results in results.items():
            msg += f'\n[{category}] Results:\n'
            for metric, value in category_results.items():
                if 'CD' in metric or 'EMD' in metric:
                    msg += f'  {metric}: {value:.6f}\n'
        
        logger.info(msg)
        
        # Save summary to main directory
        summary_file = os.path.join(self.cfg.save_dir, 'eval_summary.txt')
        with open(summary_file, 'a') as f:
            f.write(msg + '\n')
        
        return results