""" copied and modified from https://github.com/stevenygd/PointFlow/blob/master/datasets.py """
import os
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from torch.utils import data
import h5py as h5

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02747177': 'can',
    '02942699': 'camera',
    '02954340': 'cap',
    '02958343': 'car',
    '03001627': 'chair',
    '03046257': 'clock',
    '03207941': 'dishwasher',
    '03211117': 'monitor',
    '04379243': 'table',
    '04401088': 'telephone',
    '02946921': 'tin_can',
    '04460130': 'tower',
    '04468005': 'train',
    '03085013': 'keyboard',
    '03261776': 'earphone',
    '03325088': 'faucet',
    '03337140': 'file',
    '03467517': 'guitar',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03710193': 'mailbox',
    '03759954': 'microphone',
    '03761084': 'microwave',
    '03790512': 'motorcycle',
    '03797390': 'mug',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '03991062': 'pot',
    '04004475': 'printer',
    '04074963': 'remote_control',
    '04090263': 'rifle',
    '04099429': 'rocket',
    '04225987': 'skateboard',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04530566': 'vessel',
    '04554684': 'washer',
    '02992529': 'cellphone',
    '02843684': 'birdhouse',
    '02871439': 'bookshelf',
    '04574864': 'pasta15k',
    '04574865': 'pasta',
    '02858304': 'boat',
    '02834778': 'bicycle'
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

class PointClouds(Dataset):
    def __init__(self, path2data, part='train',
                 cloud_size=2**10, return_eval_cloud=False,
                 return_original_scale=False, return_bbox_scale=False,
                 cloud_transform=None):
        super().__init__()
        self.path2data = path2data
        self.part = part
        self.cloud_size = cloud_size
        self.return_eval_cloud = return_eval_cloud
        self.return_original_scale = return_original_scale
        self.return_bbox_scale = return_bbox_scale
        self.cloud_transform = cloud_transform

        self.data_file = None
        self.load_metadata()

    def load_metadata(self):
        """Load metadata arrays (bounds, centers, scales) from HDF5 file."""
        with h5.File(self.path2data, 'r', libver='latest', swmr=True) as fin:
            group = fin[self.part]
            
            self.vertices_c_bounds = np.array(group['vertices_c_bounds'][:], dtype=np.uint64)
            self.faces_bounds = np.array(group['faces_bounds'][:], dtype=np.uint64)

            if self.return_original_scale:
                self.original_centers = np.array(group['orig_c'][:], dtype=np.float32)
                self.original_scales = np.array(group['orig_s'][:], dtype=np.float32)

            if self.return_bbox_scale:
                self.bbox_centers = np.array(group['bbox_c'][:], dtype=np.float32)
                self.bbox_scales = np.array(group['bbox_s'][:], dtype=np.float32)

    def close(self):
        if self.data_file is not None:
            self.data_file.close()

    
    def sample_cloud(self, vertices_c, faces_vc, size=2**10, return_eval_cloud=False):
        polygons = vertices_c[faces_vc]
        cross = np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])
        areas = np.sqrt((cross**2).sum(1)) / 2.0

        probs = areas / areas.sum()
        p_sample = np.random.choice(np.arange(polygons.shape[0]), size=2 * size if return_eval_cloud else size, p=probs)

        sampled_polygons = polygons[p_sample]

        s1 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
        s2 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
        cond = (s1 + s2) > 1.
        s1[cond] = 1. - s1[cond]
        s2[cond] = 1. - s2[cond]

        sample = {
            'cloud': (sampled_polygons[:, 0] +
                    s1 * (sampled_polygons[:, 1] - sampled_polygons[:, 0]) +
                    s2 * (sampled_polygons[:, 2] - sampled_polygons[:, 0])).astype(np.float32)
        }

        if return_eval_cloud:
            sample['eval_cloud'] = sample['cloud'][1::2].copy().T
            sample['cloud'] = sample['cloud'][::2].T
        else:
            sample['cloud'] = sample['cloud'].T

        return sample

    def __len__(self):
        return self.vertices_c_bounds.shape[0] - 1

    def __getitem__(self, i):
        # Random seeding
        np.random.seed((i * 31 + os.getpid()) % (2**32 - 1))

        if self.data_file is None:
            self.data_file = h5.File(self.path2data, 'r', libver='latest', swmr=True)

        group = self.data_file[self.part]
        
        vertices_c = np.array(
            group['vertices_c'][self.vertices_c_bounds[i]:self.vertices_c_bounds[i + 1]],
            dtype=np.float32
        )
        faces_vc = np.array(
            group['faces_vc'][self.faces_bounds[i]:self.faces_bounds[i + 1]],
            dtype=np.uint32
        )

        sample = self.sample_cloud(
            vertices_c, faces_vc,
            size=self.cloud_size,
            return_eval_cloud=self.return_eval_cloud
        )

        if self.return_original_scale:
            sample['orig_c'] = self.original_centers[i]
            sample['orig_s'] = self.original_scales[i]

        if self.return_bbox_scale:
            sample['bbox_c'] = self.bbox_centers[i]
            sample['bbox_s'] = self.bbox_scales[i]

        if self.cloud_transform is not None:
            sample = self.cloud_transform(sample)

        return sample

def get_datasets(cfg, args):
    """
        cfg: config.data sub part 
    """
    logger.info(f'get_datasets: tr_sample_size={cfg.n_sample_points}, '
                f' te_sample_size={cfg.n_sample_points}; '
                )
    
    synsetid = cate_to_synsetid[cfg.categories]  # Single category supported for now

    path2data = os.path.join(cfg.data_dir, synsetid, 'dataset.h5')

    kwargs = {}
    tr_dataset = PointClouds(
        path2data=path2data,
        part='train',
        cloud_size=cfg.n_sample_points,
        return_eval_cloud=True,  # Need both cloud and eval_cloud for training
        return_original_scale=False,  # No scales needed for training
        return_bbox_scale=False,  # No scales needed for training
        cloud_transform=getattr(cfg, 'cloud_transform', None),
        **kwargs)

    eval_split = getattr(args, "eval_split", "test")
    te_dataset = PointClouds(
        path2data=path2data,
        part=eval_split,
        cloud_size=cfg.n_sample_points,
        return_eval_cloud=False,  # No eval_cloud needed for evaluation
        return_original_scale=True,  # Need scales for evaluation
        return_bbox_scale=True,  # Need scales for evaluation
        cloud_transform=None,
    )
    val_dataset = PointClouds(
        path2data=path2data,
        part='val',
        cloud_size=cfg.n_sample_points,
        return_eval_cloud=False,  # No eval_cloud needed for validation
        return_original_scale=True,  # Need scales for validation
        return_bbox_scale=True,  # Need scales for validation
        cloud_transform=None,
    )

    return tr_dataset, te_dataset, val_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset, val_dataset = get_datasets(cfg, args)
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
                                   pin_memory=False, **kwargs
                                   )
    test_loader = data.DataLoader(dataset=te_dataset,
                                  batch_size=cfg.batch_size_test,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  drop_last=False,
                                  )
    val_loader = data.DataLoader(dataset=val_dataset,
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
        "train_loader": train_loader,
        "val_loader": val_loader
    }
    return loaders