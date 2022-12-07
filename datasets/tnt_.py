import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, read_normal, read_normal_up, read_semantic

from .base import BaseDataset

def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1+c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])


    #  R_align = np.eye(3) # DEBUG
    R = (R_align @ R)
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    # center = pts3d.mean(0)
    center = poses[:, :3, 3].mean(0)
    # center = np.zeros((3))
    print(center)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)
    
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg

def center_poses(norm_poses, poses, render_path_poses=[], render_path=False):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(norm_poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate
    if render_path:
        render_last_row = np.tile(np.array([0, 0, 0, 1]), (len(render_path_poses), 1, 1)) # (N_images, 1, 4)
        render_path_homo = \
            np.concatenate([render_path_poses, render_last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    render_path_poses_centered = []
    if render_path:
        render_path_poses_centered = pose_avg_inv @ render_path_homo
        render_path_poses_centered = render_path_poses_centered[:, :3]

    return poses_centered, render_path_poses_centered, pose_avg_inv

class tntDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        def sort_key(x):
            if len(x) > 2 and x[-10] == "_":
                return x[-9:]
            return x
        
        if os.path.exists(os.path.join(root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(root_dir, 'rgb')):
            img_dir_name = 'rgb'
        img_files = sorted(os.listdir(os.path.join(root_dir, img_dir_name)), key=sort_key)
        if os.path.exists(os.path.join(root_dir, 'normal_mono')):
            normal_dir_name = 'normal_mono'
        if os.path.exists(os.path.join(root_dir, 'semantic')):
            sem_dir_name = 'semantic'
        # normal_files = sorted(os.listdir(os.path.join(root_dir, normal_dir_name)), key=sort_key)
        
        if split == 'train': prefix = '0_'
        elif split == 'val': prefix = '1_'
        elif 'Synthetic' in self.root_dir: prefix = '2_'
        elif split == 'test': prefix = '1_' # test set for real scenes
        
        imgs_ = sorted(glob.glob(os.path.join(self.root_dir, img_dir_name, '*.png')), key=sort_key)
        normals_ = sorted(glob.glob(os.path.join(self.root_dir, normal_dir_name, '*.png')), key=sort_key)
        semantics_ = sorted(glob.glob(os.path.join(self.root_dir, sem_dir_name, '*.pgm')), key=sort_key)
        poses_ = sorted(glob.glob(os.path.join(self.root_dir, 'pose', '*.txt')), key=sort_key)

        if split == 'train':
            imgs = list(set(imgs_) - set(imgs_[::8]))
            imgs = sorted(imgs, key=sort_key)
            normals = list(set(normals_) - set(normals_[::8]))
            normals = sorted(normals, key=sort_key)
            semantics = list(set(semantics_) - set(semantics_[::8]))
            semantics = sorted(semantics, key=sort_key)
            poses = list(set(poses_) - set(poses_[::8]))
            poses = sorted(poses, key=sort_key)
        elif split == 'val' or split == 'test':
            imgs = imgs_[::8]
            normals = normals_[::8]
            semantics = semantics_[::8]
            poses = poses_[::8]
        
        for img_name in img_files:
            img_file_path = os.path.join(root_dir, img_dir_name, img_name)
            img = Image.open(img_file_path)
            w, h = img.width, img.height
            break
        
        w, h = int(w*downsample), int(h*downsample)
        self.K = np.loadtxt(os.path.join(root_dir, 'intrinsics.txt'), dtype=np.float32)[:3, :3]
        self.K *= downsample
        self.K = torch.FloatTensor(self.K)
        
        self.img_wh = (w, h)
        self.directions, self.directions_xdx, self.directions_ydy = get_ray_directions(h, w, self.K, random=True, smooth_depth=True)
        
########################################################## get g.t. poses:
        
        self.has_render_path = False
        if split == "test" and not render_train:
            self.has_render_path = os.path.exists(os.path.join(root_dir, 'camera_path'))
        all_c2w = []
        for pose_fname in poses:
            pose_path = pose_fname
            #  intrin_path = path.join(root, intrin_dir_name, pose_fname)
            #  (right, down, forward)
            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            if len(cam_mtx) == 3:
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV

        c2w_f64 = torch.stack(all_c2w)
        # center = c2w_f64[:, :3, 3].mean(axis=0)
        # # radius = np.linalg.norm((c2w_f64[:, :3, 3]-center), axis=0).mean(axis=0)
        # up = -normalize(c2w_f64[:, :3, 1].sum(0))
        
        if self.has_render_path or render_train:
            print("render camera path" if not render_train else "render train interpolation")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose"))
                if x.endswith(".txt")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_c2w_f64 = torch.stack(all_render_c2w)
############ here we generate the test trajectories
############ we store the poses in render_c2w_f64
            ###### do interpolation ######
            if render_train:
                all_render_c2w_new = []
                for i, pose in enumerate(all_render_c2w):
                    if len(all_render_c2w_new) >= 600:
                        break
                    all_render_c2w_new.append(pose)
                    if i>0 and i<len(all_render_c2w)-1:
                        pose_new = (pose*3+all_render_c2w[i+1])/4
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1])/2
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1]*3)/4
                        all_render_c2w_new.append(pose_new)

                render_c2w_f64 = torch.stack(all_render_c2w_new)
        if kwargs.get('render_normal_mask', False):
            print("render up normal mask for train data!")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "pose"))
                if x.endswith(".txt") and x.startswith("0_")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_normal_c2w_f64 = torch.stack(all_render_c2w)
############################################################ normalize by camera
        
        norm_pose_files = sorted(
            os.listdir(os.path.join(root_dir, 'pose')), key=sort_key
        )
        norm_poses = np.stack(
            [
                np.loadtxt(os.path.join(root_dir, 'pose', x)).reshape(-1, 4)
                for x in norm_pose_files
            ],
            axis=0,
        )

        
        if self.has_render_path or render_train or kwargs.get('render_normal_mask', False):
            c2w_f64, render_c2w_f64, pose_avg_inv = center_poses(norm_poses[:,:3], c2w_f64[:,:3], render_c2w_f64[:,:3], True)
        else:
            c2w_f64, _, pose_avg_inv = center_poses(norm_poses[:,:3], c2w_f64[:,:3])
            
        self.up = -normalize(c2w_f64[:,:3,1].mean(0))
        print(f'up vector: {self.up}')
            
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 = np.array(render_normal_c2w_f64)
            render_normal_c2w_f64 = pose_avg_inv @ render_normal_c2w_f64
            render_normal_c2w_f64 = render_normal_c2w_f64[:, :3]
########################################################### scale the scene
        
        scale = np.linalg.norm(norm_poses[..., 3], axis=-1).mean()
        print(f"scene scale {scale}")
        c2w_f64[..., 3] /= scale
        
        if self.has_render_path or render_train:
            render_c2w_f64[..., 3] /= scale
        
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 /=scale
            
########################################################### gen rays
        classes = kwargs.get('classes', 7)
        self.imgs = imgs
        if split.startswith('train'):
            self.rays, self.normals, self.normals_up, self.labels = self.read_meta('train', imgs, poses, c2w_f64, normals, semantics, classes)
        else: # val, test
            self.rays, self.normals, self.normals_up, self.labels = self.read_meta(split, imgs, poses, c2w_f64, normals, semantics, classes)
            
            if self.has_render_path or render_train:
                self.render_path_rays = self.get_path_rays(render_c2w_f64)
            
            if kwargs.get('render_normal_mask', False):
                self.render_normal_rays = self.get_path_rays(render_normal_c2w_f64)

    def read_meta(self, split, imgs, poses, c2w_list, normals, semantics, classes=7):
        # rays = {} # {frame_idx: ray tensor}
        rays = []
        norms = []
        norms_up = []
        labels = []
        
        if split == 'train': prefix = '0_'
        elif split == 'val': prefix = '1_'
        elif 'Synthetic' in self.root_dir: prefix = '2_'
        elif split == 'test': prefix = '1_' # test set for real scenes
        
        self.poses = []
        print(f'Loading {len(imgs)} {split} images ...')
        for idx, (img, pose, norm, sem) in enumerate(tqdm(zip(imgs, poses, normals, semantics))):
            # c2w = np.loadtxt(pose)[:3]
            c2w = np.array(c2w_list[idx][:3])
            # c2w[:, 3] -= self.shift
            # c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]
            # if idx == 0:
            #     print(c2w)
            # rays_o, rays_d = \
            #     get_rays(self.directions, torch.FloatTensor(c2w))

            img = read_image(img_path=img, img_wh=self.img_wh)
            if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                # these scenes have black background, changing to white
                img[torch.all(img<=0.1, dim=-1)] = 1.0
            if img.shape[-1] == 4:
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            # rays[idx] = torch.cat([rays_o, rays_d, img], 1).cpu() # (h*w, 9)
            rays += [img]
            
            norm, norm_up = read_normal(norm_path=norm, norm_wh=self.img_wh)
            norm = norm @ c2w[:, :3].T # rotate into world coordinate
            norms += [norm]
            norms_up += [norm_up]
            
            label = read_semantic(sem_path=sem, sem_wh=self.img_wh, classes=classes)
            labels += [label]
        
        self.poses = torch.FloatTensor(np.stack(self.poses))
        
        return torch.FloatTensor(np.stack(rays)), torch.FloatTensor(np.stack(norms)), torch.FloatTensor(np.stack(norms_up)), torch.LongTensor(np.stack(labels))
    
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            # c2w = np.loadtxt(pose)[:3]
            render_c2w = np.array(c2w_list[idx][:3])
            # if idx == 0:
            #     print(render_c2w)
            # c2w[:, 3] -= self.shift
            # c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]

            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))

            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays