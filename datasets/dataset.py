""" copied and modified from https://github.com/stevenygd/PointFlow/blob/master/datasets.py """
import os
import time
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
import random
from pandas import read_csv

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691155': 'pasta',
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
    '02858304': 'boat',
    '02834778': 'bicycle',
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15kPCs(Dataset):
    def __init__(self,
                 categories=['airplane'],
                 tr_sample_size=10000,
                 superset_size=10000,
                 split='train',
                 scale=1.,
                 normalize_per_shape=False,
                 normalize_shape_box=False,
                 random_subsample=False,
                 sample_with_replacement=1,
                 normalize_std_per_axis=False,
                 normalize_global=False,
                 recenter_per_shape=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3,
                 return_superset=False, 
                 ):
        
        self.normalize_shape_box = normalize_shape_box
        root_dir = get_path('ShapeNet')
        self.root_dir = root_dir
        logger.info('[DATA] category: {}, split: {}, full path: {}; norm global={}, norm-box={}',
                    categories, split, self.root_dir, normalize_global, normalize_shape_box)

        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.superset_size = superset_size
        self.return_superset = return_superset
        if type(categories) is str:
            categories = [categories]
        self.cates = categories

        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]
        subdirs = self.synset_ids
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.root_dir = root_dir
        self.split = split
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.sample_with_replacement = sample_with_replacement
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        tic = time.time()
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                logger.error("Directory missing : %s " % (sub_path))
                raise ValueError('Check the data path')

            if True:
                all_mids = []
                assert(os.path.exists(sub_path)), f'path missing: {sub_path}'
                for x in os.listdir(sub_path):
                    if not x.endswith('.npy'):
                        continue
                    all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

                logger.info('[DATA] number of file [{}] under: {} ',
                            len(os.listdir(sub_path)), sub_path)
                # NOTE: [mid] contains the split: i.e. "train/<mid>"
                # or "val/<mid>" or "test/<mid>"
                all_mids = sorted(all_mids)
                for mid in all_mids:
                    obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                    self.all_points.append(point_cloud[np.newaxis, ...])
                    self.cate_idx_lst.append(cate_idx)
                    self.all_cate_mids.append((subd, mid))

        logger.info('[DATA] Load data time: {:.1f}s | dir: {} | '
                    'sample_with_replacement: {}; num points: {}', time.time() - tic, self.subdirs,
                    self.sample_with_replacement, len(self.all_points))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if self.normalize_shape_box:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (  # B,1,3
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)) / 2
            self.all_points_std = np.amax(  # B,1,1
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, input_dim)
            logger.info('all_points shape: {}. mean over axis=1',
                        self.all_points.shape)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        elif all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1,
                                                           input_dim)) / 2
            self.all_points_std = np.amax(
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        # else:  # normalize across the dataset
        elif normalize_global:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)

            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(
                    axis=0).reshape(1, 1, 1)

            logger.info('[DATA] normalize_global: mean={}, std={}',
                        self.all_points_mean.reshape(-1),
                        self.all_points_std.reshape(-1))
        else:
            raise NotImplementedError('No Normalization')
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        logger.info('[DATA] shape={}, all_points_mean:={}, std={}, max={:.3f}, min={:.3f}; num-pts={}',
                    self.all_points.shape,
                    self.all_points_mean.shape, self.all_points_std.shape,
                    self.all_points.max(), self.all_points.min(), tr_sample_size)

        if self.all_points.shape[1] < self.tr_sample_size:
            self.tr_sample_size = self.all_points.shape[1]
            logger.warning('[DATA] Training sample size reduced to {} due to insufficient points.',
                           self.tr_sample_size)
        if self.all_points.shape[1] < self.superset_size:
            self.superset_size = min(self.all_points.shape[1], self.superset_size)
            logger.warning('[DATA] Superset size reduced to {} due to insufficient points.',
                           self.superset_size)
            
        if self.superset_size < self.tr_sample_size:
            logger.warning('[DATA] Superset size {} is less than training sample size {}, ',
                           self.superset_size, self.tr_sample_size)
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape or self.normalize_shape_box:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
            self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        output = {}
        tr_out = self.all_points[idx]
        superset = self.all_points[idx]
        if self.random_subsample and self.sample_with_replacement:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
            superset_idx = np.random.choice(superset.shape[0], self.superset_size)
        elif self.random_subsample and not self.sample_with_replacement:
            tr_idxs = np.random.permutation(
                np.arange(tr_out.shape[0]))[:self.tr_sample_size]
            superset_idx = np.random.permutation(
                np.arange(superset.shape[0]))[:self.superset_size]
        else:
            tr_idxs = np.arange(self.tr_sample_size)
            superset_idx = np.arange(self.superset_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        m, s = self.get_pc_stats(idx)

        sid, mid = self.all_cate_mids[idx]
    
        output.update(
            {
                'tr_points': tr_out,
                'mean': m,
                'std': s,
                'sid': sid,
                'mid': mid,
            })
        if self.return_superset:
            superset = torch.from_numpy(superset[superset_idx, :]).float()
            output['superset'] = superset

        return output
    
class SurrogateDataset(Dataset):
    def __init__(self,
                 categories=['airplane'],
                 tr_sample_size=10000,
                 split='train',
                 scale=1.,
                 normalize_per_shape=False,
                 normalize_shape_box=False,
                 random_subsample=False,
                 sample_with_replacement=1,
                 normalize_std_per_axis=False,
                 normalize_global=False,
                 recenter_per_shape=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3,
                 return_superset=False, 
                 ):
        
        self.normalize_shape_box = normalize_shape_box
        root_dir = get_path('Surrogate')
        self.root_dir = root_dir
        logger.info('[DATA] category: {}, split: {}, full path: {}; norm global={}, norm-box={}',
                    categories, split, self.root_dir, normalize_global, normalize_shape_box)

        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.return_superset = return_superset
        if type(categories) is str:
            categories = [categories]
        self.cates = categories

        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]
        subdirs = self.synset_ids
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.root_dir = root_dir
        self.split = split
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.sample_with_replacement = sample_with_replacement
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        self.all_scores = []
        tic = time.time()
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                logger.error("Directory missing : %s " % (sub_path))
                raise ValueError('Check the data path')

            all_mids = []
            all_adhesion = []
            assert(os.path.exists(sub_path)), f'path missing: {sub_path}'
            for x in os.listdir(sub_path):
                sample_path = os.path.join(sub_path, x)
                for data_file in os.listdir(sample_path):
                    if data_file.endswith('.npy'):
                        all_mids.append(os.path.join(self.split, x[:-len('.npy')]))
                    elif data_file.endswith('.csv'):
                        all_adhesion.append(data_file[:-len('.csv')])

            logger.info('[DATA] number of file [{}] under: {} ',
                        len(os.listdir(sub_path)), sub_path)
            # NOTE: [mid] contains the split: i.e. "train/<mid>"
            # or "val/<mid>" or "test/<mid>"
            all_mids = sorted(all_mids)
            for mid in all_mids:
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                point_cloud = np.load(obj_fname)  # (15k, 3)
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

            for adhesion in all_adhesion:
                csv_fname = os.path.join(root_dir, subd, adhesion + ".csv")
                df = read_csv(csv_fname) # (15k, 3)
                self.all_scores.append(df['adhesion'].iloc[-1])

        logger.info('[DATA] Load data time: {:.1f}s | dir: {} | '
                    'sample_with_replacement: {}; num points: {}', time.time() - tic, self.subdirs,
                    self.sample_with_replacement, len(self.all_points))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]
        self.all_scores = [self.all_scores[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if self.normalize_shape_box:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (  # B,1,3
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)) / 2
            self.all_points_std = np.amax(  # B,1,1
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, input_dim)
            logger.info('all_points shape: {}. mean over axis=1',
                        self.all_points.shape)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        elif all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1,
                                                           input_dim)) / 2
            self.all_points_std = np.amax(
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        # else:  # normalize across the dataset
        elif normalize_global:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)

            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(
                    axis=0).reshape(1, 1, 1)

            logger.info('[DATA] normalize_global: mean={}, std={}',
                        self.all_points_mean.reshape(-1),
                        self.all_points_std.reshape(-1))
        else:
            raise NotImplementedError('No Normalization')
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        logger.info('[DATA] shape={}, all_points_mean:={}, std={}, max={:.3f}, min={:.3f}; num-pts={}',
                    self.all_points.shape,
                    self.all_points_mean.shape, self.all_points_std.shape,
                    self.all_points.max(), self.all_points.min(), tr_sample_size)

        if self.all_points.shape[1] < self.tr_sample_size:
            self.tr_sample_size = self.all_points.shape[1]
            logger.warning('[DATA] Training sample size reduced to {} due to insufficient points.',
                           self.tr_sample_size)
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape or self.normalize_shape_box:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
            self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        output = {}
        tr_out = self.all_points[idx]
        score = self.all_scores[idx]
        if self.random_subsample and self.sample_with_replacement:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        elif self.random_subsample and not self.sample_with_replacement:
            tr_idxs = np.random.permutation(
                np.arange(tr_out.shape[0]))[:self.tr_sample_size]
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        m, s = self.get_pc_stats(idx)

        sid, mid = self.all_cate_mids[idx]
    
        output.update(
            {
                'tr_points': tr_out,
                'mean': m,
                'std': s,
                'sid': sid,
                'mid': mid,
                'score': score,
            })

        return output

def get_path(dataname=None):
    dataset_path = {}
    dataset_path['ShapeNet'] = [
        './data/ShapeNetCore.v2.PC15k/'
    ]
    dataset_path['Surrogate'] = [
        './data/SurrogateData/'
    ]

    if dataname is None:
        return dataset_path
    else:
        assert(
            dataname in dataset_path), f'Not found {dataname}, only: {list(dataset_path.keys())}'
        for p in dataset_path[dataname]:
            if os.path.exists(p):
                return p
        raise ValueError(
            f'All path not found for {dataname}, please double check: {dataset_path[dataname]}; or edit the datasets/dataset.py Line 308')