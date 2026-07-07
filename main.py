"""Standalone sampling / inference entry point for DELIGHT.

Loads a trained checkpoint, draws samples, and writes them to disk.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch
from loguru import logger

from default_config import cfg as config


def get_args():
    parser = argparse.ArgumentParser(
        description='Sample point clouds from a trained DELIGHT model.')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to a trained checkpoint (.pt/.pth). Use a '
                             'prior checkpoint for real generation.')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file; defaults to <ckpt_dir>/../config.yml')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Only needed if the prior checkpoint predates the '
                             'VAE weights being stored inside it.')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='How many shapes to generate')
    parser.add_argument('--num_points', type=int, default=None,
                        help='Points per shape; defaults to cfg.data.n_sample_points')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Generation batch size')
    parser.add_argument('--out', type=str, default=None,
                        help='Output .pt path; defaults to <exp_dir>/samples/samples_<ts>.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', default=False, action='store_true',
                        help='Also save a png preview grid next to the samples')
    parser.add_argument('--metrics', default=False, action='store_true',
                        help='Also compute MMD/COV/1-NNA/JSD against the eval split')
    parser.add_argument('--eval_split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Split used as the reference set for --metrics')
    parser.add_argument('--denormalize', default=True, action='store_true',
                        help='Undo the dataset normalisation so samples come '
                             'out in world scale (default: on)')
    parser.add_argument('--raw', dest='denormalize', action='store_false',
                        help='Keep samples in normalised model space')
    parser.add_argument('--opt', nargs=argparse.REMAINDER, default=None,
                        help='Config overrides, e.g. --opt prior.solver rk4')

    args = parser.parse_args()
    args.pretrained = args.pretrained or None
    args.vae_checkpoint = args.vae_checkpoint or None
    args.config = args.config or None

    if not os.path.isfile(args.pretrained):
        parser.error(f'checkpoint not found: {args.pretrained}')

    # Resolve the config that the checkpoint was trained with.
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(args.pretrained), '..', 'config.yml')
    if not os.path.isfile(args.config):
        parser.error(
            f'config not found at {args.config}; pass --config explicitly')
    logger.info('Loading config: {}', args.config)
    config.merge_from_file(args.config)
    if args.opt:
        config.merge_from_list(args.opt)

    # Attributes the trainers expect from the training entry point.
    args.local_rank = 0
    args.global_rank = 0
    args.global_size = 1
    args.num_gpus = 1
    args.distributed = False
    args.eval = True
    args.resume = False
    args.ntest = args.num_samples
    args.trainer_path = 'trainers.' + config.training.type + '_trainer'

    # Write next to the checkpoint's experiment directory unless told otherwise.
    config.save_dir = os.path.normpath(
        os.path.join(os.path.dirname(args.pretrained), '..'))

    return args, config


def build_trainer(args, cfg):
    trainer_lib = importlib.import_module(args.trainer_path)
    trainer = trainer_lib.Trainer(cfg, args)
    trainer.resume(args.pretrained, eval=True)
    return trainer


@torch.no_grad()
def generate(trainer, num_samples, num_points, batch_size):
    """Draw `num_samples` shapes in batches of at most `batch_size`."""
    chunks = []
    produced = 0
    while produced < num_samples:
        bs = min(batch_size, num_samples - produced)
        logger.info('Sampling {}/{} ...', produced + bs, num_samples)
        out = trainer.sample(num_points, bs)
        chunks.append(out.detach().cpu())
        produced += bs
    return torch.cat(chunks, dim=0)[:num_samples]


def main():
    args, cfg = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if cfg.training.type != 'prior':
        logger.warning(
            'This is a "{}" checkpoint, not a prior. Samples will be decoded '
            'from codes drawn uniformly from the SoftVQ codebooks, which is a '
            'sanity check and NOT a trained generative prior. Train the prior '
            'stage and point --pretrained at that checkpoint for generation.',
            cfg.training.type)

    trainer = build_trainer(args, cfg)

    num_points = args.num_points or cfg.data.n_sample_points
    tic = time.time()
    samples = generate(trainer, args.num_samples, num_points, args.batch_size)
    logger.info('Generated {} in {:.1f}s', tuple(samples.shape), time.time() - tic)

    # Undo the dataset normalisation so the clouds are in world scale.
    if args.denormalize:
        dataset = trainer.test_loader.dataset
        mean = torch.as_tensor(dataset.all_points_mean).float().reshape(1, 1, -1)
        std = torch.as_tensor(dataset.all_points_std).float().reshape(1, 1, -1)
        if mean.shape[0] == 1:  # global stats only; per-shape stats are ambiguous
            samples = samples * std + mean
            logger.info('Denormalised with global mean={} std={}',
                        mean.reshape(-1).tolist(), std.reshape(-1).tolist())
        else:
            logger.warning('Dataset uses per-shape normalisation; generated '
                           'shapes have no matching stats, leaving them in '
                           'normalised space.')

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(cfg.save_dir, 'samples')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, 'samples_%s.pt' % time.strftime('%Y%m%d_%H%M%S'))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(samples, out_path)
    logger.info('Saved {} samples to {}', samples.shape[0], out_path)

    if args.render:
        import torchvision
        from utils.vis_helper import visualize_point_clouds_3d
        from utils.data_helper import normalize_point_clouds

        nvis = min(8, samples.shape[0])
        vis = normalize_point_clouds([samples[i] for i in range(nvis)])
        img = visualize_point_clouds_3d(vis, [f'sample-{i}' for i in range(nvis)])
        png_path = out_path.replace('.pt', '.png')
        torchvision.utils.save_image(torch.as_tensor(img) / 255.0, png_path)
        logger.info('Saved preview to {}', png_path)

    if args.metrics:
        from utils.eval_helper import compute_score

        refs = []
        for batch in trainer.test_loader:
            refs.append(batch['tr_points'])
            if sum(r.shape[0] for r in refs) >= samples.shape[0]:
                break
        ref_pcs = torch.cat(refs, dim=0)[:samples.shape[0]]
        if args.denormalize:
            dataset = trainer.test_loader.dataset
            mean = torch.as_tensor(dataset.all_points_mean).float().reshape(1, 1, -1)
            std = torch.as_tensor(dataset.all_points_std).float().reshape(1, 1, -1)
            if mean.shape[0] == 1:
                ref_pcs = ref_pcs * std + mean

        logger.info('Computing metrics against {} reference shapes ...',
                    ref_pcs.shape[0])
        results = compute_score(
            samples[:, :, :3].float(), ref_pcs[:, :, :3].float(),
            batch_size_test=min(50, args.batch_size),
            device_str='cuda' if torch.cuda.is_available() else 'cpu',
            accelerated_cd=True,
            save_vis_path=out_path.replace('.pt', '_eval.png') if args.render else None,
            visualize=args.render,
        )
        metrics_path = out_path.replace('.pt', '_metrics.txt')
        with open(metrics_path, 'w') as f:
            for k, v in results.items():
                line = f'{k}: {v:.6f}'
                logger.info(line)
                f.write(line + '\n')
        logger.info('Saved metrics to {}', metrics_path)

    logger.info('DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('Sampling failed')
        sys.exit(1)
