import os
import time
import torch
import torchvision
from abc import ABC, abstractmethod
import numpy as np
from loguru import logger

from datasets import dataset
from utils.utils import AverageMeter
from utils.vis_helper import visualize_point_clouds_3d
from utils.data_helper import normalize_point_clouds


class BaseTrainer(ABC):
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.device = torch.device('cuda:%d' % args.rank)

    @abstractmethod
    def train_iter(self, batch, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def set_writer(self, writer):
        self.writer = writer

    def epoch_end(self, epoch, step):
        if hasattr(self, 'writer'):
            self.writer.add_scalar('epoch', epoch, step)
        else:
            logger.warning("Writer not set. Skipping logging of epoch end.")

    def save(self, epoch=None, step=None, save_dir=None):
        data = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        save_dir = self.cfg.save_dir if save_dir is None else save_dir
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(save_dir, "checkpoints", save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(data, path)

        return path

    def build_data(self):
        logger.info('Building data loader...')
        train_loader, test_loader = dataset.get_data_loaders(self.cfg, self.args)

        return train_loader, test_loader
    
    def train_epochs(self):
        cfg, args = self.cfg, self.args
        writer = self.writer
        train_loader = self.train_loader

        logger.info('[GPU {}] Starting training for {} epochs'.format(args.rank, cfg.epochs))

        tic_global = time.time()
        if args.rank == 0:
            tic_log = time.time()
            start_time = time.time()
        avg_time = AverageMeter()
        step = 0

        self.total_iter = cfg.training.epochs * len(train_loader)
        
        for epoch in range(self.start_epoch, cfg.training.epochs + 1):
            self.model.train()

            if args.rank == 0:
                tic_epo = time.time()
            epoch_loss = []
           
            for idx, batch in enumerate(train_loader):
                step = idx + len(train_loader) * epoch

                if args.rank == 0 and self.writer is not None:
                    tic_iter = time.time()

                logs_info = self.train_iter(batch, step=step)

                if args.rank == 0:
                    epoch_loss.append(logs_info['loss'])

                if self.args.rank == 0 and (
                        time.time() - tic_log > 60
                ):  # log per min
                    logger.info(
                        f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.2f} | '
                        f'[exp] {cfg.save_dir} | [step] {step:5d}'
                    )
                    tic_log = time.time()

                # -- visualize rec and samples -- #
                if step % int(cfg.log_freq) == 0 and args.rank == 0 and not step == 0:
                    avg_loss = np.array(epoch_loss).mean()
                    epoch_loss = []  # clean up epoch loss
                    self.log_loss({'epo_loss': avg_loss},
                                  writer=writer, step=step)
                    visualize = int(cfg.viz_freq) > 0 and \
                        (step) % int(cfg.viz_freq) == 0
                    if visualize:
                        self.vis_recont(logs_info, writer, step)
                        self.model.eval()
                        self.vis_sample(writer, step=step,
                                        include_pred_x0=False)
                        self.model.train()

                # -- timer -- #
                if args.rank == 0 and self.writer is not None:
                    time_iter = time.time() - tic_iter
                    self.writer.avg_meter('time_iter', time_iter, step=step)

            if args.rank == 0 and self.writer is not None:
                epo_time = (time.time() - tic_epo) / 60.0  # min
                avg_time.update(epo_time)
                logger.info(
                    f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.2f} | '
                    f'[exp] {cfg.save_dir} | [step] {step:5d} | [time] {epo_time:.1f}m (~{int(avg_time.avg * (cfg.trainer.epochs - epoch) / 60)}h) | '
                    f'[best] {self.best_eval_epoch} {self.best_eval_score * 1e2:.3f}x1e-2'
                )
                tic_log = time.time()

            if epoch % int(cfg.save_freq) == 0 and int(cfg.save_freq) > 0 and args.rank == 0:
                save_path = self.save(epoch=epoch, step=step, save_dir=cfg.save_dir)
                logger.info(f"Checkpoint saved at {save_path} [Epoch]")
                self.save(epoch=epoch, step=step)
            
            if (time.time() - tic_global) / 60 > cfg.save_time and args.rank == 0:
                save_path = self.save(epoch=epoch, step=step)
                logger.info(f"Checkpoint saved at {save_path}, [Time] {(time.time() - start_time) / 60}h")
                tic_global = time.time()

            if int(cfg.val_freq) > 0 and epoch % int(cfg.val_freq) == 0 and args.rank == 0:
                score = self.evaluate(epoch=epoch, step=step)
                if score < self.best_eval_score or self.best_eval_score < 0:
                    self.save(save_name='best_eval.pth',  # save_dir=snapshot_dir,
                              epoch=epoch, step=step)
                    self.best_eval_score = score
                    self.best_eval_epoch = epoch

            self.epoch_end(epoch, step=step)

        if args.rank == 0:
            logger.info(f'Training finished, total time: {(time.time() - start_time) / 60:.2f} min')
            logger.info(f'Best eval score: {self.best_eval_score * 1e2:.3f}x1e-2 at epoch {self.best_eval_epoch}')
            self.writer.close() if hasattr(self, 'writer') else None

    @torch.no_grad()
    def vis_recont(self, logs_info, writer=None, step=None):
        """ Visualize reconstruction results """
        
        input = logs_info.get('x_0', None)
        output = logs_info.get('x_0_pred', None)

        assert len(input.shape) == len(output.shape) == 3 # (B, Npoints, 3)
        assert input.shape[0] == output.shape[0]  # batch size should match

        nvis = min(max(input.shape[0], 2), 5) # visualize at most 5, at least 2 samples

        img_list = []
        for b in range(nvis):
            x_list, name_list = [], []
            x_list.append(output[b])
            name_list.append('pred')

            x_list.append(input[b])
            name_list.append('target')

            for k, v in logs_info.items():
                if 'vis/' in k:
                    x_list.append(v[b])
                    name_list.append(k)

            x_list = normalize_point_clouds(x_list)

            vis_order = [2, 0, 1]
            vis_args = {'vis_order': vis_order}

            img = visualize_point_clouds_3d(x_list, name_list, **vis_args)

            img_list.append(img)

        img_list = torchvision.utils.make_grid(
            [torch.as_tensor(a) for a in img_list], pad_value=0)
        
        writer.add_image('vis_out/recont-train', img_list, step)