import torch
import glob
import numpy as np
import os
from PIL import Image
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import json
from .ray_utils import *
from .color_utils import read_image, read_normal, read_normal_up, read_semantic
import OpenEXR
import Imath

from .base import BaseDataset


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


class HM3DABODataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.split = split
        self.kwargs = kwargs
        prefix = self.get_split_prefix(split)

        img_paths = list(sorted((Path(root_dir) / "rgb").glob(prefix+"*.jpg")))
        h, w = self.get_img_wh(downsample, img_paths)

        depth_paths = []
        if kwargs.get('depth_mono', False):
            depth_paths = list(sorted((Path(root_dir) / "depth").glob(prefix+"*.exr")))

        # for old_path in list(sorted((Path(root_dir) / "mask").glob("*.png"))):
        #     os.rename(str(old_path), str(Path(root_dir) / "mask" / ('0_' + old_path.name)))

        target_c2w_f64, target_indices = self.get_target_poses(prefix, root_dir)
        center, scale = self.get_pose_scale_and_center(root_dir)
        # target_c2w_f64[:, :3, 3] -= center
        target_c2w_f64[..., 3] /= scale

        self.get_intrinsics(root_dir, downsample)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
        
########################################################## get g.t. poses:
        
        self.has_render_traj = False
        if split == "test" and not render_train:
            self.has_render_traj = True # os.path.exists(os.path.join(root_dir, 'camera_path'))

###########################################################
        if self.has_render_traj or render_train:
            print("render camera path" if not render_train else "render train interpolation")
            test_c2w_f64 = self.get_target_poses("", root_dir)[0]
            # test_c2w_f64[:, :3, 3] -= center
            test_c2w_f64[..., 3] /= scale
    
    ############ here we generate the test trajectories
    ############ we store the poses in render_c2w_f64
            ###### do interpolation ######
            if render_train:
                all_render_c2w_new = []
                for i, pose in enumerate(test_c2w_f64):
                    if len(all_render_c2w_new) >= 600:
                        break
                    all_render_c2w_new.append(pose)
                    if i>0 and i<len(test_c2w_f64)-1:
                        pose_new = (pose*3+test_c2w_f64[i+1])/4
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+test_c2w_f64[i+1])/2
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+test_c2w_f64[i+1]*3)/4
                        all_render_c2w_new.append(pose_new)

                test_c2w_f64 = torch.stack(all_render_c2w_new)
            self.render_traj_rays = self.get_path_rays(test_c2w_f64)

########################################################### gen rays

        if split.startswith('train'):
            # dino features
            target_dino_features = self.load_dino_features(root_dir, target_indices)

            # clip features
            target_clip_features = self.load_clip_features(root_dir, target_indices,
                                                           target_shape=target_dino_features.shape[2:])

            self.rays, self.poses, self.meta_features = self.read_meta('train', img_paths, target_c2w_f64,
                                                       target_dino_features, target_clip_features)
            if len(depth_paths) > 0:
                self.depths_2d = self.read_depth(depth_paths) / scale
        else: # val, test
            # dino features
            target_dino_features = self.load_dino_features(root_dir, target_indices)

            # clip features
            target_clip_features = self.load_clip_features(root_dir, target_indices,
                                                           target_shape=target_dino_features.shape[2:])

            self.rays, self.poses, self.meta_features = self.read_meta(split, img_paths, target_c2w_f64,
                                                       target_dino_features, target_clip_features)
            if len(depth_paths)>0:
                self.depths_2d = self.read_depth(depth_paths) / scale

    def load_dino_features(self, root_dir, target_indices):
        target_dino_features = []
        for dino_path in list(sorted((Path(root_dir) / "dino_features").glob("*.npy"))):
            index = int(dino_path.name[:-4])
            if index in target_indices:
                target_dino_features.append(torch.from_numpy(np.load(str(dino_path))))
        target_dino_features = torch.stack(target_dino_features)
        return target_dino_features

    def load_clip_features(self, root_dir, target_indices, target_shape):
        target_clip_features = {}  # patch_size -> features
        for clip_path in list(sorted((Path(root_dir) / "clip_features").glob("*.npy"))):
            patch_size = int(clip_path.name[-7:-4]) / 480
            index = int(clip_path.name.split("_")[0])
            if patch_size not in target_clip_features:
                target_clip_features[patch_size] = []
            if index in target_indices:
                clip_feature = torch.from_numpy(np.load(str(clip_path))).float()
                target_clip_features[patch_size].append(clip_feature)

        for key in target_clip_features.keys():
            target_clip_features[key] = torch.stack(target_clip_features[key]).permute(0, 3, 1, 2)
            target_clip_features[key] = torch.nn.functional.interpolate(target_clip_features[key], size=target_shape, mode="bilinear")

        return target_clip_features

    def get_split_prefix(self, split):
        if split == 'train':
            prefix = '0_'
        elif split == 'val':
            prefix = '1_'
        elif split == 'test':
            prefix = '1_'  # test set for real scenes
        else:
            prefix = ''
        return prefix

    def get_img_wh(self, downsample, img_paths):
        img = Image.open(str(img_paths[0]))
        w, h = img.width, img.height
        w, h = int(w * downsample), int(h * downsample)
        self.img_wh = (w, h)
        return h, w

    def get_intrinsics(self, root_dir, downsample):
        with open(root_dir + "/intrinsics.txt", "r") as f:
            K = []
            for line in f.readlines():
                K.append([float(num) for num in line.strip().split(" ")])
            self.K = np.array(K)
        if self.K.shape[0]>4:
            self.K = self.K.reshape((4, 4))
        self.K = self.K[:3, :3]
        self.K *= downsample
        self.K = torch.FloatTensor(self.K)

    def get_target_poses(self, prefix, root_dir):
        target_poses = []
        target_indices = []
        for pose_fn in sorted((Path(root_dir) / "pose").glob(prefix + "*.txt")):
            cam_mtx = np.loadtxt(pose_fn).reshape(-1, 4)
            target_poses.append(torch.from_numpy(cam_mtx))
            if prefix == pose_fn.name[:2]:
                target_indices.append(int(pose_fn.name.split('_')[1][:-4]))
        target_c2w_f64 = torch.stack(target_poses)
        return target_c2w_f64, target_indices

    def get_pose_scale_and_center(self, root_dir):
        all_poses = []
        for pose_fn in sorted((Path(root_dir) / "pose").glob("*.txt")):
            cam_mtx = np.loadtxt(pose_fn).reshape(-1, 4)
            all_poses.append(cam_mtx)
        center = np.stack(all_poses)[:, :3, 3].mean(axis=0)
        scale = np.linalg.norm(np.stack(all_poses)[..., 3], axis=-1).max()
        print(f"{self.split} scene scale {scale} center {center}")
        return center, scale

    def read_meta(self, split, imgs, c2w_list, target_dino_features, target_clip_features):
        # rays = {} # {frame_idx: ray tensor}
        rays = []
        poses = []
        print(f'Loading {len(imgs)} {split} images ...')
        for idx, img in enumerate(tqdm(imgs)):
            c2w = np.array(c2w_list[idx][:3])
            poses += [c2w]
            img = read_image(img_path=img, img_wh=self.img_wh)
            if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                # these scenes have black background, changing to white
                img[torch.all(img <= 0.1, dim=-1)] = 1.0
            if img.shape[-1] == 4:
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            rays += [img]
        # kwargs.get('use_sem', False)
        metadata = {}
        if self.kwargs.get('use_clip', False):
            metadata['clip_patch_scales'] = np.array(sorted(list(target_clip_features.keys())))
            for k in target_clip_features.keys():
                target_clip_features[k] = target_clip_features[k].permute(0, 2, 3, 1)
                target_clip_features[k] = target_clip_features[k].reshape(target_clip_features[k].shape[0], -1, target_clip_features[k].shape[-1])
            metadata['clip'] = torch.stack([target_clip_features[k] for k in metadata['clip_patch_scales']])
        if self.kwargs.get('use_dino', False):
            target_dino_features = target_dino_features.permute(0, 2, 3, 1)
            target_dino_features = target_dino_features.reshape(target_dino_features.shape[0], -1, target_dino_features.shape[-1])
            metadata['dino'] = target_dino_features

        return torch.FloatTensor(np.stack(rays)), torch.FloatTensor(np.stack(poses)), metadata
    
    def read_depth(self, depths):
        depths_ = []

        for depth in depths:
            im = OpenEXR.InputFile(str(depth))
            dw = im.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            depth = np.frombuffer(im.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)),
                                  dtype=np.float32)
            depth.shape = (size[1], size[0])

            depths_ += [rearrange(depth, 'h w -> (h w)')]
        return torch.FloatTensor(np.stack(depths_))
    
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])

            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))

            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays
