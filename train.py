from loguru import logger
import importlib
import argparse
import os
import sys
import time
import torch.multiprocessing as mp
from default_config import cfg as config

from utils import utils

@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def main(args, config):
    logger.info('Training: {}', config.training.type)

    trainer_lib = importlib.import_module(args.trainer_path)
    Trainer = trainer_lib.Trainer
    trainer = Trainer(config, args)

    writer = utils.init(args.global_rank, save_dir=config.save_dir)

    if args.global_rank == 0:
        trainer.set_writer(writer)

    ckpt_dir = os.path.join(config.save_dir, 'checkpoints')
    snapshot_file = os.path.join(config.save_dir, 'checkpoints', 'snapshot.pth')

    # -- check if prev saved ckpt exist -- #
    if os.path.exists(ckpt_dir) and os.path.exists(snapshot_file):
        logger.info(
            '[Detect saved snapshot at the checkpoint dir] resume from preemption!!! ')
        args.resume = True
        args.pretrained = os.path.join(
            config.save_dir, 'checkpoints', 'snapshot.pth')
    else:
        if args.global_rank == 0:
            logger.info('Could not find any checkpoint: {}, (exist={}), or snapshot {}, (exist={})',
                        ckpt_dir, os.path.exists(ckpt_dir), snapshot_file, os.path.exists(snapshot_file))

    if args.resume or args.eval:
        assert args.pretrained is not None, "Pretrained model path must be provided for resuming."
        trainer.resume(args.pretrained, eval=args.eval)
    elif args.pretrained is not None:
        logger.info('Resuming training from {}; if you do not want to resume training, edit the config to change the exp name',
                    args.pretrained)
        trainer.resume(args.pretrained)

    if not args.eval:
        trainer.train_epochs()
    else:
        if not args.skip_nll:
            trainer.eval_nll(trainer.step, ntest=args.ntest, save_file=True)
        # vis sampled output
        if not args.skip_sample:
            trainer.vis_sample(writer=trainer.writer,
                               step=trainer.step)
            trainer.eval_sample(trainer.step)
        logger.info('DONE')
        

def get_args():
    parser = argparse.ArgumentParser(description='Train or evaluate the VAE model.')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--exp_root', type=str, default='../experiments')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to the pretrained model for resuming training or evaluation')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--skip_sample', type=int, default=0,
                        help='only eval nll, no sampling')
    parser.add_argument('--skip_nll', type=int, default=0,
                        help='skip eval nll ')
    parser.add_argument('--opt',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='Address of the master node for distributed training')
    parser.add_argument('--ntest', type=int, default=100,
                        help='Number of test samples for evaluation')

    args = parser.parse_args()

    args.trainer_path = "trainers." + config.training.type + "_trainer"

    if args.eval or args.resume:
        logger.info('Arguments: {}'.format(args))
        args.config = os.path.dirname(args.pretrained) + '/../cfg.yaml'
        config.merge_from_file(args.config)

    config.merge_from_list(args.opt)

    if config.exp_name == '' or config.exp_name is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Build detailed experiment name
        exp_components = [
            config.data.categories,
            f"bs{config.data.batch_size}",
            timestamp
        ]

        config.exp_name = "_".join(exp_components)

    logger.info(f'Generated experiment name: {config.exp_name}')

    if args.eval:
        # For evaluation, use the existing experiment directory but add eval suffix
        base_dir = os.path.dirname(args.config) if hasattr(args, 'config') else os.path.join(args.exp_root, config.exp_name)
        eval_suffix = f"_eval_{time.strftime('%Y%m%d_%H%M%S')}" if not hasattr(args, 'config') else "_eval"
        config.save_dir = config.log_dir = config.log_name = base_dir + eval_suffix
    else:
        # For training, create organized directory structure
        if config.training.type == 'ddpm':
            base_exp_dir = os.path.join(args.exp_root, config.training.type)
        else:
            base_exp_dir = os.path.join(args.exp_root, config.training.type + '_' + config.model.quantizer)
        config.log_name = os.path.join(base_exp_dir, config.exp_name)
        config.save_dir = os.path.join(base_exp_dir, config.exp_name)  
        config.log_dir = os.path.join(base_exp_dir, config.exp_name)
    
    os.makedirs(config.log_dir, exist_ok=True)

    # save config and log
    if args.global_rank == 0 and not args.eval:
        logger.add(config.log_dir + '/train.log')
        logger.info('Exp root: {} + exp name: {}, save dir: {}', args.exp_root,
                    config.exp_name, config.save_dir)
        saved_cfg = os.path.join(config.log_dir, 'config.yml')
        with open(saved_cfg, 'w') as file:
            file.write(config.dump())
        logger.info('Save config at {}', saved_cfg)
        
        # Also save a human-readable experiment info file
        exp_info_file = os.path.join(config.log_dir, 'experiment_info.txt')
        with open(exp_info_file, 'w') as f:
            f.write(f"Experiment Name: {config.exp_name}\n")
            f.write(f"Dataset: {config.data.categories}\n")
            f.write(f"Training Type: {config.training.type}\n")
            f.write(f"Batch Size: {config.data.batch_size}\n")
            f.write(f"Learning Rate: {config.training.opt.lr}\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command Line Args: {' '.join(sys.argv)}\n")
            f.write(f"Latent Dimension: {config.model.latent_dim}\n")
        logger.info('Save experiment info at {}', exp_info_file)

    elif args.eval:
        logger.add(config.log_dir + '/eval_gen.log')
    logger.info('Log dir: {}', config.log_dir)

    return args, config

def main_worker(local_rank, args, config):
    """Worker function for each GPU process"""
    args.local_rank = local_rank
    args.global_rank = local_rank + args.node_rank * args.num_gpus
    args.global_size = args.num_proc_node * args.num_gpus
    args.distributed = args.num_gpus > 1

    logger.info(f'Node rank {args.node_rank}, local proc {local_rank}, global proc {args.global_rank}')

    # Initialize distributed training if needed
    if args.distributed:
        utils.init_processes(args.global_rank, args.global_size, args)

    # Run the main training/evaluation logic
    main(args, config)


if __name__ == "__main__":    
    args, config = get_args()

    if args.num_gpus > 1:
        mp.set_start_method('spawn', force=True)
        mp.spawn(fn=main_worker, args=(args, config), nprocs=args.num_gpus)
    else:
        main_worker(local_rank=0, args=args, config=config)
    
    logger.info('Training or evaluation completed.')