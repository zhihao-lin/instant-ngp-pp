import numpy as np
import os 
import cv2
import pandas as pd
import json
from PIL import Image
import math
import utm
import pvlib
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

from .ray_utils import *
from .base import BaseDataset

class HighbayDataset(BaseDataset):
    def __init__(self, root_dir, split, nvs=False, downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        self.nvs = nvs # exclude testing frames in training

        dir_rgb_0 = os.path.join(root_dir, 'left', 'rgb')
        dir_rgb_1 = os.path.join(root_dir, 'right', 'rgb')
        dir_sem_0 = os.path.join(root_dir, 'left', 'semantic')
        dir_sem_1 = os.path.join(root_dir, 'right', 'semantic')
        dir_normal_0 = os.path.join(root_dir, 'left', 'normal')
        dir_normal_1 = os.path.join(root_dir, 'right', 'normal')
        path_csv = os.path.join(root_dir, 'gps.csv')
        sensor_data = pd.read_csv(path_csv)

        # Intrinsics
        intrinsic_path = os.path.join(root_dir, 'transforms.json')
        with open(intrinsic_path, 'r') as file:
            intrinsic = json.load(file)
        K = np.array([
            [intrinsic['fl_x'], 0, intrinsic['cx']],
            [0, intrinsic['fl_y'], intrinsic['cy'] ],
            [0, 0, 1]
        ])
        K[:2] *= downsample
        self.K = K
        w, h = int(intrinsic['w']), int(intrinsic['h'])
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

        # Extrinsics
        valid_name = kwargs.get('valid_id', 'valid.txt')
        valid_path = os.path.join(root_dir, valid_name)
        img_time = np.load(os.path.join(root_dir, 'img_time.npy'))
        valid_ids, valid_time  = self.get_valid_time(img_time, valid_path)
        self.setup_poses(sensor_data, valid_time)
        print('#frames = {}'.format(len(valid_time)))
        
        print('Load RGB ...')
        rgb_0 = self.read_rgb(dir_rgb_0, valid_ids)
        rgb_1 = self.read_rgb(dir_rgb_1, valid_ids)
        self.rays = torch.FloatTensor(np.concatenate([rgb_0, rgb_1], axis=0))
        if self.split == 'train':
            print('Load Semantic ...')
            sem_0 = self.read_semantics(dir_sem_0, valid_ids)
            sem_1 = self.read_semantics(dir_sem_1, valid_ids)
            self.labels = torch.LongTensor(np.concatenate([sem_0, sem_1], axis=0))
            print('Load Normal ...')
            normal_0 = self.read_normal(dir_normal_0, valid_ids)
            normal_1 = self.read_normal(dir_normal_1, valid_ids)
            self.normals = torch.FloatTensor(np.concatenate([normal_0, normal_1], axis=0))

    def get_valid_time(self, img_time, valid_path):
        valids = []
        with open(valid_path) as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                v = line.strip().split(',')
                v = [int(i) for i in v]
                valids.append(v)

        valid = img_time < 0
        for start, end in valids:
            sample = (img_time >= start) & (img_time <= end)
            valid = valid | sample
        valid_ids = np.arange(len(img_time))
        valid_ids = valid_ids[valid]
        valid_time = img_time[valid]
        return valid_ids, valid_time

    def setup_poses(self, sensor_data, valid_time):
        sensor_time = np.array(sensor_data['field.header.stamp'])
        latitude    = np.array(sensor_data['field.latitude'])
        longitude   = np.array(sensor_data['field.longitude'])
        height      = np.array(sensor_data['field.height'])
        roll        = np.array(sensor_data['field.roll'])
        pitch       = np.array(sensor_data['field.pitch'])
        azimuth     = np.array(sensor_data['field.azimuth'])
        
        euler = np.stack([pitch, roll, -azimuth]).T
        r = R.from_euler('xyz', euler, degrees=True)
        rot = r.as_matrix()#.transpose(0, 2, 1)  # (n, 3, 3)
        to_cv = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        rot = rot @ to_cv
        Rs = R.from_matrix(rot)
        f_rot = Slerp(sensor_time, Rs)
        rot_sample = f_rot(valid_time)
        rot_sample = np.stack([r.as_matrix() for r in rot_sample])

        utm_data = utm.from_latlon(latitude, longitude)
        east, north = utm_data[0], utm_data[1]
        pos = np.stack([east, north, height]) #(3, n)
        f_pos = interp1d(sensor_time, pos)
        pos_sample = f_pos(valid_time).T 
        pt_min = np.min(pos_sample, axis=0)
        pt_max = np.max(pos_sample, axis=0)
        center = (pt_min + pt_max) / 2
        scale  = np.max(pt_max - pt_min) / 2
        pos_sample = (pos_sample - center[None]) / scale # normalize to cube within (-1, 1)

        c2w_l = np.zeros((len(pos_sample), 3, 4))
        c2w_l[:, :3, :3] = rot_sample
        c2w_l[:, :3, -1] = pos_sample
        c2w_r = np.zeros((len(pos_sample), 3, 4))
        c2w_r[:, :3, :3] = rot_sample
        x = rot_sample[:, :, 0]
        c2w_r[:, :3, -1] = pos_sample + x * 0.12 / scale # lens distance = 120 mm = 0.12 m 
        c2w = np.concatenate([c2w_l, c2w_r], axis=0)
        self.poses = torch.FloatTensor(c2w)

        if self.split != 'train':
            render_c2w = generate_interpolated_path(c2w, 5)[:400]
            self.render_c2w = torch.FloatTensor(render_c2w)
            self.render_traj_rays = self.get_path_rays(render_c2w)

    def get_path_rays(self, render_c2w):
        rays = {}
        print(f'Loading {len(render_c2w)} camera path ...')
        for idx in range(len(render_c2w)):
            c2w = np.array(render_c2w[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

    def read_rgb(self, dir_rgb, valid_ids):
        rgb_list = []
        for i in valid_ids:
            path = os.path.join(dir_rgb, '{:0>5d}.png'.format(i))
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32)
            rays = img.reshape(-1, 3)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list
    
    def read_semantics(self, dir_sem, valid_ids):
        label_list = []
        for i in valid_ids:
            path = os.path.join(dir_sem, '{:0>5d}.pgm'.format(i))
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = label.flatten()
            label_list.append(label)
        label_list = np.stack(label_list)
        return label_list
    
    def read_normal(self, dir_normal, valid_ids):
        poses = self.poses.numpy()
        normal_list = []
        for c2w, i in zip(poses, valid_ids):
            path = os.path.join(dir_normal, '{:0>5d}_normal.npy'.format(i))
            img = np.load(path).transpose(1, 2, 0)
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list


def test():
    kwargs = {
    }
    dataset = HighbayDataset(
        '/hdd/datasets/highbay/0904/8_28_23_2023-09-04-09-13-16',
        'train', nvs=False,
        ** kwargs
    )
    dataset.ray_sampling_strategy = 'all_images'
    dataset.batch_size = 256
    print('poses:', dataset.poses.size())
    print('RGBs: ', dataset.rays.size())

    sample = dataset[0]
    print('Keys:')
    print(sample.keys())

    import open3d 
    import vedo
    vizualizer = open3d.visualization.Visualizer()
    vizualizer.create_window()
    vizualizer.create_window(width=1280, height=720)
    w, h = dataset.img_wh
    K = dataset.K
    poses = dataset.poses.numpy()
    poses[:, :, -1] *= 100
    # for i in range(len(poses)):
    #     pose = np.concatenate([poses[i], np.array([[0, 0, 0, 1]])], axis=0)
    #     pose = np.linalg.inv(pose)
    #     cam = open3d.geometry.LineSet.create_camera_visualization(view_width_px=w, view_height_px=h, intrinsic=K, extrinsic=pose)
    #     vizualizer.add_geometry(cam)
    # vizualizer.run()
    pos = poses[:, :, -1]
    arrow_len, s = 1, 1
    x_end   = pos + arrow_len * poses[:, :, 0]
    y_end   = pos + arrow_len * poses[:, :, 1]
    z_end   = pos + arrow_len * poses[:, :, 2]
    
    x = vedo.Arrows(pos, x_end, s=s, c='red')
    y = vedo.Arrows(pos, y_end, s=s, c='green')
    z = vedo.Arrows(pos, z_end, s=s, c='blue')
        
    vedo.show(x,y,z, axes=1)

def test2():
    dir_img = '/hdd/datasets/highbay/0904/8_28_23_2023-09-04-09-13-16/img_l'
    names = sorted([name for name in os.listdir(dir_img)])
    ids_img   = np.array([int(name.split('.')[0]) for name in names])
    
    gps_path = '/hdd/datasets/highbay/0904/8_28_23_2023-09-04-09-13-16/gps_sync.csv'
    data = pd.read_csv(gps_path)
    stamp = np.array(data['field.header.stamp'])
    
    print(ids_img[1])
    print(stamp[1])
    print(ids_img[1] - stamp[1])

if __name__ == '__main__':
    test()