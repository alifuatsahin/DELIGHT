from loguru import logger
import argparse
import os
import sys

from utils import utils

@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def main(args, config):
    writer = utils.init(args.global_rank, config.save_dir)

    writer.add_hparams(config.to_dict(), vars(args))

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')

    if os.path.exists(ckpt_path):
        logger.info('Found existing checkpoint, loading...')
    else:
        logger.info('No checkpoint found, starting fresh.')

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