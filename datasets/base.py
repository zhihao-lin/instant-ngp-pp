import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
                img_idxs = (np.ones(self.batch_size) * img_idxs).astype(np.int64)
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            if hasattr(self, 'img_wh'):
                w, h = self.img_wh
                u = pix_idxs // w
                v = pix_idxs % w
                uv = np.stack([u, v], axis=-1)
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs, 'uv': uv,
                    'rgb': rays[:, :3]}
            if hasattr(self, 'labels'):
                labels = self.labels[img_idxs, pix_idxs]
                sample['label'] = labels

            if hasattr(self, 'meta_features'):
                uu = np.round(u/(480/60))
                vv = np.round((v - 80)/(480/60))
                mm = (uu >= 0) * (uu < 60) * (vv >= 0) * (vv < 60)
                pp = ((vv + uu * 60) * mm).astype(np.int)
                sample["feature_mask"] = mm
                if getattr(self, 'kwargs')['use_dino']:
                    sample["dino"] = self.meta_features['dino'][img_idxs, pp]
                if getattr(self, 'kwargs')['use_clip']:
                    patch_scales = np.random.randint(0, len(self.meta_features['clip_patch_scales']), size=(pp.shape[0],))
                    sample['clip_patch_scales'] = self.meta_features["clip_patch_scales"][patch_scales]
                    sample["clip"] = self.meta_features["clip"][patch_scales, img_idxs, pp]


            if hasattr(self, 'depths_2d'):
                depth = self.depths_2d[img_idxs, pix_idxs]
                sample['depth'] = depth
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if hasattr(self, 'labels'):
                    labels = self.labels[idx]
                    sample['label'] = labels
                if hasattr(self, 'depths_2d'):
                    depth = self.depths_2d[idx]
                    sample['depth'] = depth
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
            if hasattr(self, 'meta_features'):
                if getattr(self, 'kwargs')['use_dino']:
                    sample["dino"] = self.meta_features['dino'][idx]
                if getattr(self, 'kwargs')['use_clip']:
                    sample["clip"] = dict((patch_ratio, clip_feature[idx]) for patch_ratio, clip_feature in self.meta_features['clip'].items())
        return sample