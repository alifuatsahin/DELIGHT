from .dataset import Uniform15kPCs

from loguru import logger
from torch.utils import data

def get_datasets(cfg, args, return_superset=False):
    """
        cfg: config.data sub part 
    """
    logger.info(f' get_datasets: tr_sample_size={cfg.n_sample_points}, '
                f' superset_size={cfg.superset_size}; '
                f' random_subsample={cfg.random_subsample}'
                f' normalize_global={cfg.normalize_global}'
                f' normalize_std_per_axis={cfg.normalize_std_per_axis}'
                f' normalize_per_shape={cfg.normalize_per_shape}'
                f' recenter_per_shape={cfg.recenter_per_shape}'
                )
    tr_dataset = Uniform15kPCs(
        categories=cfg.categories,
        split='train',
        tr_sample_size=cfg.n_sample_points,
        superset_size=cfg.superset_size,
        sample_with_replacement=cfg.sample_with_replacement,
        scale=cfg.dataset_scale,  # root_dir=cfg.data_dir,
        normalize_shape_box=cfg.normalize_shape_box,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        normalize_global=cfg.normalize_global,
        recenter_per_shape=cfg.recenter_per_shape,
        random_subsample=cfg.random_subsample,
        return_superset=return_superset,
    )

    eval_split = getattr(args, "eval_split", "val")
    # te_dataset has random_subsample as False, therefore not using sample_with_replacement
    te_dataset = Uniform15kPCs(
        categories=cfg.categories,
        split=eval_split,
        tr_sample_size=cfg.n_sample_points,
        superset_size=cfg.superset_size,
        scale=cfg.dataset_scale,  # root_dir=cfg.data_dir,
        normalize_shape_box=cfg.normalize_shape_box,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        normalize_global=cfg.normalize_global,
        recenter_per_shape=cfg.recenter_per_shape,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        return_superset=False, # For evaluation superset is not needed
    )
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args, return_superset=False):
    tr_dataset, te_dataset = get_datasets(cfg, args, return_superset=return_superset)
    kwargs = {}
    if args.distributed:
        kwargs['sampler'] = data.distributed.DistributedSampler(
            tr_dataset, shuffle=True)
    else:
        kwargs['shuffle'] = True
    if args.eval:
        kwargs['shuffle'] = False
    train_loader = data.DataLoader(dataset=tr_dataset,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.num_workers,
                                   drop_last=cfg.train_drop_last == 1,
                                   pin_memory=False, **kwargs)
    test_loader = data.DataLoader(dataset=te_dataset,
                                  batch_size=cfg.batch_size_test,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  drop_last=False,
                                  )
    logger.info(
        f'[Batch Size] train={cfg.batch_size}, test={cfg.batch_size_test}; drop-last={cfg.train_drop_last}')
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders