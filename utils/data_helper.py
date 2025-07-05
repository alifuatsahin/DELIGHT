import numpy as np
from scipy.spatial.transform import Rotation
from torchvision.transforms import Compose


class Scale2OrigCloud(object):
    def __init__(self, **kwargs):
        self.do_rescale = kwargs['cloud_rescale2orig']
        self.do_recenter = kwargs['cloud_recenter2orig']

    def __call__(self, sample):
        if self.do_rescale:
            sample['cloud'] = sample['orig_s'] * sample['cloud']
            if 'eval_cloud' in sample:
                sample['eval_cloud'] = sample['orig_s'] * sample['eval_cloud']
        if self.do_recenter:
            sample['cloud'] = sample['cloud'] + sample['orig_c'].reshape(-1, 1)
            if 'eval_cloud' in sample:
                sample['eval_cloud'] = sample['eval_cloud'] + sample['orig_c'].reshape(-1, 1)
        return sample


class TranslateCloud(object):
    def __init__(self, **kwargs):
        self.shift = np.array(kwargs['cloud_translate_shift'], dtype=np.float32).reshape(-1, 1)

    def __call__(self, sample):
        sample['cloud'] -= self.shift
        if 'eval_cloud' in sample:
            sample['eval_cloud'] -= self.shift
        return sample


class ScaleCloud(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('cloud_scale_scale'))

    def __call__(self, sample):
        sample['cloud'] /= self.scale
        if 'eval_cloud' in sample:
            sample['eval_cloud'] /= self.scale
        return sample


class AddNoise2Cloud(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('cloud_noise_scale'))

    def __call__(self, sample):
        sample['cloud'] += np.random.normal(scale=self.scale, size=sample['cloud'].shape).astype(np.float32)
        if 'eval_cloud' in sample:
            sample['eval_cloud'] += np.random.normal(scale=self.scale, size=sample['eval_cloud'].shape).astype(np.float32)
        return sample


class CenterCloud(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['cloud'] -= sample['cloud'].mean(axis=1, keepdims=True)
        if 'eval_cloud' in sample:
            sample['eval_cloud'] -= sample['eval_cloud'].mean(axis=1, keepdims=True)
        return sample


class Random3DRotation(object):

    def __call__(self, sample):
        random_3d_rotation = Rotation.random()
        sample['cloud'] = np.transpose(
            random_3d_rotation.apply(np.transpose(sample['cloud'], (1, 0))), (1, 0)).astype(np.float32)
        sample['eval_cloud'] = np.transpose(
            random_3d_rotation.apply(np.transpose(sample['cloud'], (1, 0))), (1, 0)).astype(np.float32)
        sample['rotation'] = np.tile(random_3d_rotation.as_euler('zxy', degrees=False), (1, 1)).astype(np.float32)
        return sample


def ComposeCloudTransformation(**kwargs):
    cloud_transformation = []
    cloud_transformation_val = []
    if kwargs.get('cloud_rescale2orig') or kwargs.get('cloud_recenter2orig'):
        cloud_transformation.append(Scale2OrigCloud(**kwargs))
        cloud_transformation_val.append(Scale2OrigCloud(**kwargs))
    if kwargs.get('cloud_translate'):
        cloud_transformation.append(TranslateCloud(**kwargs))
        cloud_transformation_val.append(TranslateCloud(**kwargs))
    if kwargs.get('cloud_scale'):
        cloud_transformation.append(ScaleCloud(**kwargs))
        cloud_transformation_val.append(ScaleCloud(**kwargs))
    if kwargs.get('cloud_noise'):
        cloud_transformation.append(AddNoise2Cloud(**kwargs))
        cloud_transformation_val.append(AddNoise2Cloud(**kwargs))
    if kwargs.get('cloud_center'):
        cloud_transformation.append(CenterCloud())
        cloud_transformation_val.append(CenterCloud())
    if kwargs.get('cloud_random_rotate'):
        cloud_transformation.append(Random3DRotation())

    if len(cloud_transformation) == 0:
        return None, None
    else:
        return Compose(cloud_transformation), Compose(cloud_transformation_val),

def normalize_point_clouds(pc_list):
    """
    Normalize a point cloud to fit within [-1, 1].
    """
    assert type(pc_list) is list, f'expect list, get {type(pc_list)}'
    output_list = []

    for pc in pc_list:
        pc = pc_list[i]
        pc = pc.detach().clone()

        assert len(pc.shape) == 2, f'expect 2D tensor, get {pc.shape}'

        pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
        pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)

        pc_min = pc_min[:, :3] # Take only the first 3 dimensions in case there are other features
        pc_max = pc_max[:, :3]

        shift = ((pc_min + pc_max) / 2).view(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc[:, :3] = (pc[:, :3] - shift) / scale
        # pcs[i] = pc
        output_list.append(pc)

    return output_list


