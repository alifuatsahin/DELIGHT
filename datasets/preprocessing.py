from modules.otplan import OTPlanSampler
from tqdm import tqdm
from loguru import logger
import torch
import os

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
    '02858304': 'boat',
    '02834778': 'bicycle',
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

BATCH_SIZE = 10
DATA_PATH = "data/ShapeNetCore.v2.PC15k"

def sample_noise_like(x):
    return torch.randn_like(x)

def build_ot_plan(cat='all', superset_size=30000):
    pcs = {}
    ot_sampler = OTPlanSampler(p=2, blur=0.05)
    for cat in os.listdir(DATA_PATH):
        subdir = os.path.join(DATA_PATH, cat)
        if not os.path.isdir(subdir):
            continue
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(subdir, split)
            if not os.path.isdir(split_dir):
                continue
            for point_cloud_file in tqdm(os.listdir(split_dir)):
                x1 = torch.load(os.path.join(split_dir, point_cloud_file))
                pcs['x1'].append(x1)
                pcs['labels'].append(point_cloud_file)

    logger.info("[DATA] Loaded successfully.")

    x1 = torch.stack(pcs['x1']) # (B, N, D)
    B, N, D = x1.shape
    if superset_size > N:
        logger.warning(f"Superset size {superset_size} is smaller than point cloud size {N}. Adjusting to {N}.")
        superset_size = N
    x0 = torch.randn(B, superset_size, D)
