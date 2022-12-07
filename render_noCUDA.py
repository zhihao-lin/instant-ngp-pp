import os
import torch
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import numpy as np
import cv2
import math
from tqdm import trange
from models.network_distill import NGP_distill
from models.networks_noCUDA import NGP, Normal
from models.networks_prop import NGP_prop
from models.rendering_noCUDA import render
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

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model_prop = NGP_prop(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    models = [model_prop, model]
    normal_model = Normal().cuda()
    if split=='train':
        ckpt_path = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs+hparams.normal_epochs-1}_slim.ckpt'
    else:
        ckpt_path = hparams.ckpt_path        
    print(f'ckpt specified: {ckpt_path} !')
    load_ckpt(model_prop, ckpt_path, model_name='model_prop', prefixes_to_ignore=['embedding_a', 'normal_net.params'])
    load_ckpt(model, ckpt_path, prefixes_to_ignore=['prop', 'embedding_a', 'normal_net.params'])
    load_ckpt(normal_model, ckpt_path, model_name='normal_model', prefixes_to_ignore=["embedding_a", "center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params"])
    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'

    if hparams.dataset_name=='tnt':
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        N_imgs = N_imgs - math.ceil(N_imgs/8)
        embed_a_length = hparams.embed_a_len
        if hparams.embed_a:
            embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])
            embedding_a = embedding_a(torch.tensor([0]).cuda())        
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train}
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

    frame_series = []
    frame_up_series = []
    depth_series = []
    normal_series = []
    semantic_series = []
    composite_series = []
    for img_idx in trange(len(render_path_rays)):
        rays = render_path_rays[img_idx][:, :6].cuda()
        render_kwargs = {'test_time': True,
                    'T_threshold': 1e-2,
                    'use_skybox': hparams.use_skybox,
                    'render_rgb': hparams.render_rgb,
                    'render_depth': hparams.render_depth,
                    'render_normal': hparams.render_normal,
                    'render_up_sem': hparams.render_normal_up,
                    'render_sem': hparams.render_semantic,
                    'img_wh': dataset.img_wh,
                    'samples': hparams.samples,
                    'num_classes': hparams.num_classes}
        if hparams.dataset_name in ['colmap', 'nerfpp']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a0'] = embedding_a
            render_kwargs['embedding_a1'] = embedding_a
        # results = render(models, rays[:, :3], rays[:, 3:6], **render_kwargs)
        chunk_size = 8192
        results = {}
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        for i in range(0, rays_o.shape[0], chunk_size):
            ret = render(models, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **render_kwargs)
            for k in ret:
                if k not in results:
                    results[k] = []
                results[k].append(ret[k])
        for k in results:
            if k in ['total_samples']:
                continue
            results[k] = torch.cat(results[k], 0)
        results['total_samples'] = torch.sum(torch.tensor(results['total_samples']))
        
        if hparams.render_rgb:
            rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_frame = (rgb_frame*255).astype(np.uint8)
            frame_series.append(rgb_frame)
            if img_idx%10 == 0:
                imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_rgb{img_idx}.png'), rgb_frame)
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
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_series.append(depth)
            # imageio.imsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', f'test_depth{img_idx}.png'), depth)
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