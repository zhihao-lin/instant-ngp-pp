import torch
from torch import nn
from opt import get_opts
import os
import glob
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import numpy as np
import cv2
import random
import math
from einops import rearrange

# data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
# from models.networks import NGP, Normal
from models.networks_sem_4 import NGP, Normal
from models.rendering_vertical import render, MAX_SAMPLES
from models.global_var import global_var

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

# render path
from tqdm import trange
from utils import load_ckpt
from render_watersurface import render_for_test
import trimesh

from torch import autograd

import warnings; warnings.filterwarnings("ignore")


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


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False
                        
        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.normal_model = None
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        if hparams.normal_distillation:
            self.normal_model = Normal()
        # if hparams.ckpt_path is not None:
        #     print(f'Check point specified: {hparams.ckpt_path}')
        #     load_ckpt(self.model, hparams.ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net.params'])
            
        ###
        self.N_imgs = 0
        embed_a_length = hparams.embed_a_len
        if hparams.dataset_name == 'tnt':
            if os.path.exists(os.path.join(hparams.root_dir, 'images')):
                img_dir_name = 'images'
            elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
                img_dir_name = 'rgb'
            
            self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
            # self.N_imgs = self.N_imgs - math.ceil(self.N_imgs/8)
            if hparams.embed_a:
                self.embedding_a = torch.nn.Embedding(self.N_imgs, embed_a_length)  
                # if hparams.ckpt_path is not None:
                #     load_ckpt(self.embedding_a, hparams.ckpt_path, model_name='embedding_a', \
                #     prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])   
        ###
        
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        self.surface_points = []
        self.surface_color = []

    def forward(self, batch, split):
        # import ipdb; ipdb.set_trace()
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions
        
        poses_ = poses

        if self.hparams.embed_a and split=='train':
            embedding_a = self.embedding_a(batch['img_idxs']).detach()
        elif self.hparams.embed_a and split=='test':
            # idx = batch['img_idxs']*7
            # if idx>=len(self.train_dataset.imgs):
            #     idx = len(self.train_dataset.imgs)-1
            embedding_a = self.embedding_a(torch.tensor([0], device=directions.device)).detach()

        rays_o, rays_d = get_rays(directions, poses_)
        
        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox if self.global_step>=self.warmup_steps else False,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_up_sem': hparams.render_normal_up,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'up': self.up,
                  'ground_height': self.ground_height}
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        if self.hparams.embed_a:
            kwargs['embedding_a'] = embedding_a
        
        if split == 'train':
            return render(self.model, rays_o, rays_d, **kwargs)
        else:
            # import ipdb; ipdb.set_trace()
            chunk_size = 8192*16
            # w, h = rays_o.shape[0], rays_o.shape[1]
            # rays_o_flat = rays_o.reshape(-1, 3)
            # rays_d_flat = rays_d.reshape(-1, 3)
            all_ret = {}
            for i in range(0, rays_o.shape[0], chunk_size):
                ret = render(self.model, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])
            # import ipdb; ipdb.set_trace()
            for k in all_ret:
                if k in ['total_samples']:
                    continue
                all_ret[k] = torch.cat(all_ret[k], 0)
            # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret and k not in ['total_samples']}
            all_ret['total_samples'] = torch.sum(torch.tensor(all_ret['total_samples']))
            return all_ret
                

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'use_sem': self.hparams.render_semantic,
                  'use_upLabel': self.hparams.render_normal_up}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        
        self.img_wh = self.test_dataset.img_wh

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        self.register_parameter('up', nn.Parameter(self.train_dataset.up.to(self.device)))
        self.register_parameter('ground_height', nn.Parameter(torch.zeros(1, device=self.device)))
                
        load_ckpt(self.model, self.hparams.ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net'])
        load_ckpt(self.embedding_a, self.hparams.ckpt_path, model_name='embedding_a', prefixes_to_ignore=['model', 'normal_net'])
        
        net_params = []
        for n, p in self.named_parameters():
            if n in ['up', 'ground_height']: net_params += [p]
        
        opts = []
        # up_opt = Adam([self.up], 1e-6)
        # height_opt = Adam([self.ground_height], 1e-4)
        # opts += [up_opt, height_opt]
        
        net_opt = Adam([{'params': [self.up], 'lr': 1e-5},
                        {'params': [self.ground_height], 'lr': 1e-3}])
        opts = [net_opt]
        
        # net_sch = CosineAnnealingLR(self.net_opt,
        #                             self.hparams.num_epochs,
        #                             self.hparams.lr/30)       

        return opts

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    # def on_train_start(self):
    #     self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
    #                                     self.poses,
    #                                     self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        tensorboard = self.logger.experiment            

        results = self(batch, split='train')
        
        # import ipdb; ipdb.set_trace()
        up = F.normalize(self.up, dim=0)
        cos_theta = torch.sum(results['normal']*up[None, :], dim=1,)
        # weight = torch.exp(1-results['depth']/results['depth'].min()) * torch.exp(cos_theta.detach()-1)
        weight = torch.exp(cos_theta.detach()-1)
        loss_n = torch.mean(weight[:, None]*(results['normal']-up)**2)
        
        loss_v = torch.mean(results['vertical_dist']**2 * torch.exp(-results['vertical_dist'].detach()*10))
        loss = 0.1*loss_n + loss_v
        
        # self.surface_points.append(results['xyz'].cpu().numpy())
        # self.surface_color.append(results['rgb'].cpu().numpy())

        self.log('train/loss', loss)
        self.log('train/up[0]', up[0].detach().cpu())
        self.log('train/up[1]', up[1].detach().cpu())
        self.log('train/up[2]', up[2].detach().cpu())
        self.log('train/ground_height', self.ground_height[0].detach().cpu())
        # self.log('train/grads_inf_cnt', results['gradinf_cnt'])
        # self.log('train/Ro', results['Ro'].mean())
        # self.log('train/Rp', results['Rp'].mean())
        if self.global_step%5000 == 0 and self.global_step>0:
            print('[val in training]')
            # import ipdb; ipdb.set_trace()
            # batch = self.test_dataset[random.randint(0, math.ceil(self.N_imgs/8)-1)]
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            tensorboard.add_image('img/render0', rgb_pred.cpu().numpy(), self.global_step)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            tensorboard.add_image('img/depth0', depth_pred, self.global_step)
            batch = self.test_dataset[8]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            tensorboard.add_image('img/render1', rgb_pred.cpu().numpy(), self.global_step)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            tensorboard.add_image('img/depth1', depth_pred, self.global_step)
            batch = self.test_dataset[13]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            tensorboard.add_image('img/render2', rgb_pred.cpu().numpy(), self.global_step)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            tensorboard.add_image('img/depth2', depth_pred, self.global_step)
            # batch = self.test_dataset[12]
            # for i in batch:
            #     if isinstance(batch[i], torch.Tensor):
            #         batch[i] = batch[i].cuda()
            # results = self(batch, split='test')
            # w, h = self.img_wh
            # rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            # tensorboard.add_image('img/render12', rgb_pred.cpu().numpy(), self.global_step)
            # self.surface_points = np.concatenate(self.surface_points, axis=0)
            # self.surface_color = np.concatenate(self.surface_color, axis=0)
            # surface_color = np.ones((self.surface_color.shape[0], 4))
            # surface_color[:, :3] = self.surface_color
            
            # point_cloud = trimesh.points.PointCloud(self.surface_points, surface_color)
            # point_cloud.export(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}/test_surface_{self.global_step}.ply'))
            
            # self.surface_points = []
            # self.surface_color = []
            
        # import ipdb; ipdb.set_trace()
        return loss
        
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    torch.manual_seed(20220806)
    torch.cuda.manual_seed_all(20220806)
    np.random.seed(20220806)
    hparams = get_opts()
    autograd.set_detect_anomaly(True)
    global_var._init()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    if hparams.normal_distillation_only:
        assert hparams.weight_path is not None, "No weight ckpt specified when distilling normals"
        hparams.num_epochs = 0
    if not hparams.normal_distillation:
        hparams.normal_epochs = 0
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.normal_epochs + hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)
    
    trainer = Trainer(max_epochs=hparams.num_epochs+hparams.normal_epochs,
                      check_val_every_n_epoch=0,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0,
                      precision=32,
                      gradient_clip_val=50,
                      detect_anomaly=True)

    trainer.fit(system)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs+hparams.normal_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs+hparams.normal_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
        
    render_for_test(hparams, split='train')
