import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
import random
import math
from pathlib import Path
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.implicit_mask import implicit_mask
from models.rendering import render, MAX_SAMPLES
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
from render import render_for_test
import trimesh
from kornia import create_meshgrid

from torch import autograd

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth, scale=16):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def mask2img(mask):
    mask_img = cv2.applyColorMap((mask*255).astype(np.uint8),
                                  cv2.COLORMAP_BONE)

    return mask_img

def semantic2img(sem_label, classes):
    level = 1/(classes-1)
    sem_color = level * sem_label
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
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        if hparams.embed_msk:
            self.msk_model = implicit_mask()
            
        ### setup appearance embeddings
        img_dir_name = None
        if os.path.exists(os.path.join(hparams.root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
            img_dir_name = 'rgb'

        if hparams.dataset_name == 'kitti':
            self.N_imgs = 2 * hparams.train_frames
        elif hparams.dataset_name == 'mega':
            self.N_imgs = 1920 // 6
        elif hparams.dataset_name == 'crop':
            self.N_imgs = len(list(Path(hparams.root_dir).glob("im*")))
        elif hparams.dataset_name == 'hm3d_abo':
            self.N_imgs = len(list((Path(hparams.root_dir) / img_dir_name).glob("0_*")))
        else:
            self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        
        if hparams.embed_a:
            self.embedding_a = torch.nn.Embedding(self.N_imgs, hparams.embed_a_len) 
        ###
        
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions
        
        poses_ = poses
        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses_ = torch.zeros_like(poses).cuda()
            poses_[..., :3] = dR @ poses[..., :3]
            dT = self.dT[batch['img_idxs']]
            poses_[..., 3] = poses[..., 3] + dT

        if self.hparams.embed_a and split=='train':
            embedding_a = self.embedding_a(batch['img_idxs'])
        elif self.hparams.embed_a and split=='test':
            embedding_a = self.embedding_a(torch.tensor([0], device=directions.device))

        rays_o, rays_d = get_rays(directions, poses_)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox if self.global_step>=self.warmup_steps else False,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh}
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        if self.hparams.embed_a:
            kwargs['embedding_a'] = embedding_a
        
        if split == 'train':
            return render(self.model, rays_o, rays_d, **kwargs)
        else:
            chunk_size = self.hparams.chunk_size
            all_ret = {}
            for i in range(0, rays_o.shape[0], chunk_size):
                ret = render(self.model, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])
            for k in all_ret:
                if k in ['total_samples']:
                    continue
                all_ret[k] = torch.cat(all_ret[k], 0)
            all_ret['total_samples'] = torch.sum(torch.tensor(all_ret['total_samples']))
            return all_ret
                

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'use_sem': self.hparams.render_semantic,
                  'depth_mono': self.hparams.depth_mono}

        if self.hparams.dataset_name == 'kitti':
            kwargs['scene'] = self.hparams.kitti_scene
            kwargs['start'] = self.hparams.start
            kwargs['train_frames'] = self.hparams.train_frames
            center_pose = []
            for i in self.hparams.center_pose:
                center_pose.append(float(i))
            val_list = []
            for i in self.hparams.val_list:
                val_list.append(int(i))
            kwargs['center_pose'] = center_pose
            kwargs['val_list'] = val_list
        if self.hparams.dataset_name == 'mega':
            kwargs['mega_frame_start'] = self.hparams.mega_frame_start
            kwargs['mega_frame_end'] = self.hparams.mega_frame_end

        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        
        self.img_wh = self.test_dataset.img_wh
        
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

    def configure_optimizers(self):
        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
        
        load_ckpt(self.model, self.hparams.ckpt_load, prefixes_to_ignore=['embedding_a', 'msk_model'])
        if self.hparams.embed_a:
            load_ckpt(self.embedding_a, self.hparams.ckpt_load, model_name='embedding_a', prefixes_to_ignore=['model', 'msk_model'])
        if self.hparams.embed_msk:
            load_ckpt(self.msk_model, self.hparams.ckpt_load, model_name='msk_model', prefixes_to_ignore=['model', 'embedding_a'])

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]
        
        opts = []
        # self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-8)
        self.net_opt = Adam(net_params, self.hparams.lr, eps=1e-8)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [Adam([self.dR, self.dT], 1e-6)]
            
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs+self.hparams.normal_epochs,
                                    self.hparams.lr/30)      

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        tensorboard = self.logger.experiment
        
        uniform_density = None
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                        warmup=self.global_step<self.warmup_steps,
                                        erode=self.hparams.dataset_name=='colmap')

        # with autograd.detect_anomaly():
        results = self(batch, split='train')
        
        if self.hparams.embed_msk:
            w, h = self.img_wh
            uv = torch.tensor(batch['uv']).cuda()
            img_idx = torch.tensor(batch['img_idxs']).cuda()
            uvi = torch.zeros((uv.shape[0], 3)).cuda()
            uvi[:, 0] = (uv[:, 0]-h/2) / h
            uvi[:, 1] = (uv[:, 1]-w/2) / w
            uvi[:, 2] = (img_idx - self.N_imgs/2) / self.N_imgs
            mask = self.msk_model(uvi)
        
        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                    'uniform_density': uniform_density,
                    'normal_p': True,
                    'semantic': self.hparams.render_semantic,
                    'depth_mono': self.hparams.depth_mono,
                    'embed_msk': self.hparams.embed_msk,
                    'step': self.global_step}
        if self.hparams.embed_msk:
            loss_kwargs['mask'] = mask
        loss_d = self.loss(results, batch, **loss_kwargs)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())
        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/s_per_ray', results['total_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)
        if self.global_step%10000 == 0 and self.global_step>0:
            print('[val in training]')
            w, h = self.img_wh
            
            # uv = create_meshgrid(h, w, False, device=self.device)[0]
            # mask = []
            # chunk_size = 8192*16
            # for i in range(0, h*w, chunk_size):
            #     with torch.no_grad():
            #         uv_ = uv.reshape(-1, 2)[i:i+chunk_size]
            #         uvi = torch.zeros((uv_.shape[0], 3)).cuda()
            #         uvi[:, 0] = (uv_[:, 1]-h/2) / h
            #         uvi[:, 1] = (uv_[:, 0]-w/2) / w
            #         uvi[:, 2] = (81-self.N_imgs/2) / self.N_imgs
            #         mask_ = self.msk_model(uvi)
            #     mask.append(mask_)
            # mask = torch.cat(mask, dim=0)
            # mask_pred = mask2img(rearrange(mask.squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h))
            # mask_pred = rearrange(mask_pred, 'h w c -> c h w', h=h)
            # tensorboard.add_image('img/mask_pred', mask_pred, self.global_step)
            
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            if hparams.render_semantic:
                semantic_pred = semantic2img(rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h), self.hparams.get('classes', 7))
                semantic_pred  = rearrange(semantic_pred , 'h w c -> c h w', h=h)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h), scale=2*self.hparams.scale)
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            normal_pred = rearrange((results['normal_pred']+1)/2, '(h w) c -> c h w', h=h)
            rgb_gt = rearrange(batch['rgb'], '(h w) c -> c h w', h=h)

            tensorboard.add_image('img/render', rgb_pred.cpu().numpy(), self.global_step)
            if hparams.render_semantic:
                tensorboard.add_image('img/semantic', semantic_pred, self.global_step)
            tensorboard.add_image('img/depth', depth_pred, self.global_step)
            tensorboard.add_image('img/normal_pred', normal_pred.cpu().numpy(), self.global_step)
            tensorboard.add_image('img/gt', rgb_gt.cpu().numpy(), self.global_step)
            

        for name, params in self.model.named_parameters():
            check_nan=None
            check_inf=None
            if params.grad is not None:
                check_nan = torch.any(torch.isnan(params.grad))
                check_inf = torch.any(torch.isinf(params.grad))
            if check_inf or check_nan:
                import ipdb; ipdb.set_trace()

        return loss
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)
    
    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs
    
    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        print(f'test/mean_PSNR: {mean_psnr}')
        self.log('test/psnr', mean_psnr)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        print(f'test/mean_SSIM: {mean_ssim}')
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            print(f'test/mean_LPIPS: {mean_lpips}')
            self.log('test/lpips_vgg', mean_lpips)

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
    global_var._init()
    if hparams.val_only and (not hparams.ckpt_load):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename=hparams.ckpt_save.split('.')[0],
                              save_weights_only=True,
                              every_n_epochs=1,
                              save_last=True,
                              save_on_train_epoch_end=True)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=5,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32,
                      gradient_clip_val=50)

    trainer.fit(system)

    # save slimmed ckpt for the last epoch
    ckpt_ = slim_ckpt(os.path.join(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}', 'last.ckpt'),
            save_poses=hparams.optimize_ext)
    torch.save(ckpt_, os.path.join(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}', 'last_slim.ckpt'))

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
