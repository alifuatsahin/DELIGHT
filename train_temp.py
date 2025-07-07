from loguru import logger
import importlib
import argparse
import os
import sys

from utils import utils

@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def main(args, config):
    writer = utils.init(args.rank, config.save_dir)

    trainer_lib = importlib.import_module(config.trainer.type)
    Trainer = trainer_lib.Trainer
    trainer = Trainer(config, args)

    if args.rank == 0:
        trainer.set_writer(writer)
        if len(config.bash_name) > 0 and os.path.exists(config.bash_name):
            writer.log_asset(config.bash_name)
        if len(config.bash_name) > 0 and os.path.exists(os.path.join(config.save_dir, config.bash_name.split('/')[-1])):
            writer.log_asset(os.path.join(
                config.save_dir, config.bash_name.split('/')[-1]))
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
        logger.info('Could not find any checkpoint: {}, (exist={}), or snapshot {}, (exist={})',
                    ckpt_dir, os.path.exists(ckpt_dir), snapshot_file, os.path.exists(snapshot_file))

    if args.resume or args.eval:
        assert args.pretrained is not None, "Pretrained model path must be provided for resuming."

    if args.eval:
        logger.info('Evaluating the model...')
        trainer.eval(args.pretrained)

    else:
        logger.info('Starting training...')
        trainer.train()
        

def get_args():
    parser = argparse.ArgumentParser(description='Train or evaluate the VAE model.')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--exp_root', type=str, default='exp')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Which dataset to use')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to the pretrained model for resuming training or evaluation')
    parser.add_argument('--eval', default=False, action='store_true')

    args = parser.parse_args()

    if args.eval or args.resume:
        logger.info('Arguments: {}'.format(args))
        args.config = os.path.dirname(args.pretrained) + '/../config.yaml'
        config.merge_from_file(args.config)
    elif args.config != 'none':
        logger.info('Loading config from: {}'.format(args.config))
        config.merge_from_file(args.config)

    config.merge_from_list(args.opt)

    EXP_ROOT = args.exp_root

if __name__ == "__main__":
    args, config = get_args()

    if args.num_gpus > 1:
        args.distributed = True
        processes = []
        for rank in range(args.num_gpus):
            logger.info(f'Starting process for GPU {rank}')
            args.local_rank = rank
            p = Process(target=utils.init_processes,
                        args=(main, args, config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
    else:
        args.distributed = False
        utils.init_processes(main, args, config)
    
    logger.info('Training or evaluation completed.')