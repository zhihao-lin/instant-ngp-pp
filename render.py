import os
import torch
import imageio
import numpy as np
import cv2
import math 
from PIL import Image
from tqdm import trange
from models.network_distill import NGP_distill
from models.networks_sem_4 import NGP, Normal
from models.rendering_ import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from opt import get_opts
from einops import rearrange

def depth2img(depth):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    depth = depth/16
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def semantic2img(sem_label, classes):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    level = 1/(classes-1)
    sem_color = level * sem_label
    # depth = np.clip(depth, a_min=0., a_max=1.)
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color

def render_chunks(model, rays_o, rays_d, chunk_size, **kwargs):
    chunk_n = math.ceil(rays_o.shape[0]/chunk_size)
    results = {}
    for i in range(chunk_n):
        rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
        rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
        ret = render(model, rays_o_chunk, rays_d_chunk, **kwargs)
        for k in ret:
            if k not in results:
                results[k] = []
            results[k].append(ret[k])
    for k in results:
        if k in ['total_samples']:
            continue
        results[k] = torch.cat(results[k], 0)
    return results

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    normal_model = Normal().cuda()
    if hparams.ckpt_load:
        ckpt_path = hparams.ckpt_load
    else: 
        ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'last_slim.ckpt')

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net', 'directions', 'density_grid', 'grid_coords'])
    load_ckpt(normal_model, ckpt_path, model_name='normal_model', prefixes_to_ignore=["embedding_a", "center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params"])
    print('Loaded checkpoint: {}'.format(ckpt_path))
    
    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'
    
    if hparams.dataset_name == 'kitti':
        N_imgs = 2 * hparams.train_frames
    elif hparams.dataset_name == 'mega':
        N_imgs = 1920 // 6
    else:
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
    
    embed_a_length = hparams.embed_a_len
    if hparams.embed_a:
        embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())        
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_path': hparams.render_path,
            'climate': hparams.climate,
            'anti_aliasing_factor': hparams.anti_aliasing_factor}
    if hparams.dataset_name == 'kitti':
            kwargs['scene'] = hparams.kitti_scene
            kwargs['start'] = hparams.start
            kwargs['train_frames'] = hparams.train_frames
            center_pose = []
            for i in hparams.center_pose:
                center_pose.append(float(i))
            val_list = []
            for i in hparams.val_list:
                val_list.append(int(i))
            kwargs['center_pose'] = center_pose
            kwargs['val_list'] = val_list
    if hparams.dataset_name == 'mega':
            kwargs['mega_frame_start'] = hparams.mega_frame_start
            kwargs['mega_frame_end'] = hparams.mega_frame_end

    dataset = dataset(split='test', **kwargs)
    w, h = dataset.img_wh
    if hparams.render_path:
        render_path_rays = dataset.render_path_rays
    else:
        # render_path_rays = dataset.rays
        render_path_rays = {}
        print("generating rays' origins and directions!")
        for img_idx in trange(len(dataset.poses)):
            rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
            render_path_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/frames'
    os.makedirs(frames_dir, exist_ok=True)

    frame_series = []
    frame_up_series = []
    depth_raw_series = []
    depth_series = []
    points_series = []
    normal_series = []
    semantic_series = []
    composite_series = []
    mask_series = []

    for img_idx in trange(len(render_path_rays)):
        rays = render_path_rays[img_idx][:, :6].cuda()
        render_kwargs = {
            'img_idx': img_idx,
            'test_time': True,
            'T_threshold': 1e-2,
            'use_skybox': hparams.use_skybox,
            'render_rgb': hparams.render_rgb,
            'render_depth': hparams.render_depth,
            'render_normal': hparams.render_normal,
            'render_up_sem': hparams.render_normal_up,
            'render_sem': hparams.render_semantic,
            'distill_normal': hparams.render_normal,
            'img_wh': dataset.img_wh,
            'normal_model': normal_model,
            'anti_aliasing_factor': hparams.anti_aliasing_factor
        }
        if hparams.dataset_name in ['colmap', 'nerfpp']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        results = {}
        chunk_size = hparams.chunk_size
        if chunk_size > 0:
            results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)
        else:
            results = render(model, rays_o, rays_d, **render_kwargs)

        if hparams.render_rgb:
            rgb_frame = None
            if hparams.anti_aliasing_factor > 1.0:
                h_new = int(h*hparams.anti_aliasing_factor)
                rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h_new)
                rgb_frame = Image.fromarray((rgb_frame*255).astype(np.uint8)).convert('RGB')
                rgb_frame = np.array(rgb_frame.resize((w, h), Image.Resampling.BICUBIC))
            else:
                rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                rgb_frame = (rgb_frame*255).astype(np.uint8)
            frame_series.append(rgb_frame)
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-rgb.png'.format(img_idx)), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            # imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_rgb{img_idx}.png'), frame)
        if hparams.render_normal_up:
            up_frame = rearrange(results['rgb+up'].cpu().numpy(), '(h w) c -> h w c', h=h)
            up_frame = (up_frame*255).astype(np.uint8)
            frame_up_series.append(up_frame)
            # imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_rgb{img_idx}.png'), frame)
        if hparams.render_semantic:
            sem_frame = semantic2img(rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h), 7)
            # frame = (frame).astype(np.uint8)
            semantic_series.append(sem_frame)
            # import ipdb; ipdb.set_trace()
            # imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_semantic{img_idx}.png'), frame)
        if hparams.render_depth:
            depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
            depth_raw_series.append(depth_raw)
            depth = depth2img(depth_raw)
            depth_series.append(depth)
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-depth.png'.format(img_idx)), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
        
        if hparams.render_points:
            points = rearrange(results['points'].cpu().numpy(), '(h w) c -> h w c', h=h)
            points_series.append(points)

        if hparams.render_mask:
            mask = rearrange(results['mask'].cpu().numpy(), '(h w) -> h w', h=h)
            mask_series.append(mask)
            # cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-mask.png'.format(img_idx)), mask.astype(np.float32)*255)

        if hparams.render_normal:
            normal = rearrange(results['normal_pred'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6            
            normal_series.append((255*(normal+1)/2).astype(np.uint8))
            if img_idx%100 == 0:
                imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_normal{img_idx}.png'), normal)
        if hparams.normal_composite:
            normal = rearrange(results['normal_pred'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6            
            composite = np.ones((normal.shape[0], normal.shape[1], 3))
            up_mask = np.zeros_like(normal)
            # import ipdb; ipdb.set_trace()
            up_mask[..., :] = dataset.up
            # up_mask[..., :] = np.array([0.0039,-0.7098,-0.6941])
            valid_mask = np.linalg.norm(normal.reshape(-1, 3), axis=-1, keepdims=True)!=0
            theta_between_up = np.matmul(up_mask.reshape(-1, 3)[:, None, :], normal.reshape(-1, 3)[:, :, None]).squeeze(-1)\
                            /(np.linalg.norm(up_mask.reshape(-1, 3), axis=-1, keepdims=True)*np.linalg.norm(normal.reshape(-1, 3), axis=-1, keepdims=True)-1e-6)
            # near_up = np.logical_and(theta_between_up>0.7, theta_between_up<=1.)
            # import ipdb; ipdb.set_trace()
            near_up = np.logical_and(theta_between_up>0.866, valid_mask)
            # near_up = theta_between_up>0.3
            near_up = np.reshape(near_up, (h, w))
            # import ipdb; ipdb.set_trace()
            composite[near_up] = (255*theta_between_up.reshape(normal.shape[0], normal.shape[1], 1)*np.ones_like(composite)).astype(np.uint8)[near_up]
            composite[~near_up] = rgb_frame[~near_up]
            composite = composite.astype(np.uint8)
            composite_series.append(composite)
            if img_idx%100 == 0:
                imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_composite{img_idx}.png'), composite)
            
        torch.cuda.synchronize()

    if hparams.render_rgb:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_path.mp4' if not hparams.render_train else "circle_path.mp4"),
                        frame_series,
                        fps=30, macro_block_size=1)
    if hparams.render_normal_up:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_path_up.mp4' if not hparams.render_train else "circle_path_up.mp4"),
                        frame_up_series,
                        fps=30, macro_block_size=1)
    if hparams.render_semantic:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_semantic.mp4' if not hparams.render_train else "circle_path_semantic.mp4"),
                        semantic_series,
                        fps=30, macro_block_size=1)
    if hparams.render_depth:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_path_depth.mp4' if not hparams.render_train else "circle_path_depth.mp4"),
                        depth_series,
                        fps=30, macro_block_size=1)
        
        depth_raw_all = np.stack(depth_raw_series) #(n_frames, h ,w)
        path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw.npy'
        np.save(path, depth_raw_all)

    if hparams.render_points:
        points_all = np.stack(points_series)
        path = f'results/{hparams.dataset_name}/{hparams.exp_name}/points.npy'
        np.save(path, points_all)
    
    if hparams.render_mask:
        mask_all = np.stack(mask_series)
        path = f'results/{hparams.dataset_name}/{hparams.exp_name}/mask.npy'
        np.save(path, mask_all)

    if hparams.render_normal:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_path_normal.mp4' if not hparams.render_train else "circle_path_normal.mp4"),
                        normal_series,
                        fps=30, macro_block_size=1)
    if hparams.normal_composite:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_path_composite.mp4' if not hparams.render_train else "circle_path_composite.mp4"),
                        composite_series,
                        fps=30, macro_block_size=1)

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.normal_distillation_only:
        assert hparams.ckpt_path is not None, "No ckpt specified when distilling normals"
        hparams.num_epochs = 0
    render_for_test(hparams)