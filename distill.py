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
import math
import random
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.network_distill_sem_4 import NGP_distill
from models.rendering_ import render, MAX_SAMPLES

# optimizer, losses
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

from utils import slim_ckpt

# render path
from tqdm import trange
from utils import load_ckpt
from render import render_for_test

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16 # the interval to update density grid

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        # import ipdb; ipdb.set_trace()
        self.model = NGP_distill(scale=self.hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        assert hparams.ckpt_path is not None
        # load_ckpt(self.model, hparams.ckpt_path, prefixes_to_ignore=['dir_encoder','rgb_net','normal_net','embedding_a'])
        load_ckpt(self.model, hparams.ckpt_path, prefixes_to_ignore=['normal_net','embedding_a'])
        
        self.N_imgs = 0
        embed_a_length = hparams.embed_a_len
        if hparams.dataset_name == 'tnt':
            if os.path.exists(os.path.join(hparams.root_dir, 'images')):
                img_dir_name = 'images'
            elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
                img_dir_name = 'rgb'
            
            self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
            self.N_imgs = self.N_imgs - math.ceil(self.N_imgs/8)
            if hparams.embed_a:
                self.embedding_a = torch.nn.Embedding(self.N_imgs, embed_a_length)  
        load_ckpt(self.embedding_a, hparams.ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])
        
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        self.frame_series = []

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        elif split=='test':
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            dT = self.dT[batch['img_idxs']]
            poses[..., 3] += dT

        if self.hparams.embed_a and split=='train':
            embedding_a = self.embedding_a(batch['img_idxs'])
        elif self.hparams.embed_a and split=='test':
            idx = batch['img_idxs']*7
            if idx>=len(self.train_dataset.imgs):
                idx = len(self.train_dataset.imgs)-1
            embedding_a = self.embedding_a(torch.tensor([idx], device=directions.device))
        elif self.hparams.embed_a and split=='val':
            embedding_a = self.embedding_a(torch.tensor([0]).cuda())
        
        if split in ['train', 'test']:
            rays_o, rays_d = get_rays(directions, poses)
        else:
            # len_ = len(self.test_dataset.render_path_rays)
            # img_idx = len_*self.global_step//(self.hparams.num_epochs*1000)
            rays = self.test_dataset.render_path_rays[self.global_step][:, :6].cuda()
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
        
        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': False,
                  'distill_normal': False,
                  'img_wh': self.img_wh,
                  'normal_model': False,
                  'depth_smooth': False}
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
                  'render_train': self.hparams.render_train,
                  'render_path': self.hparams.render_path}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        
        self.img_wh = self.test_dataset.img_wh

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT', 'embedding_a.weight']: net_params += [p]
            
        embeda_params = []
        for n, p in self.embedding_a.named_parameters():
            embeda_params += [p]
        opts = []
        # self.net_opt = Adam(net_params, self.hparams.lr, eps=1e-15)
        self.net_opt = Adam([{'params': net_params}, 
                          {'params': embeda_params, 'lr': 1e-6}], lr=self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            # learning rate is hard-coded
            pose_r_opt = Adam([self.dR], 1e-6)
            pose_t_opt = Adam([self.dT], 1e-6)
            opts += [pose_r_opt, pose_t_opt]
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
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

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        tensorboard = self.logger.experiment
        # if self.global_step%self.update_interval == 0:
        #     self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
        #                                     warmup=self.global_step<self.warmup_steps,
        #                                     erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                       'up_sem': False,
                       'distill': True}
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
        if self.global_step%5000 == 0 and self.global_step>0:
            print('[val in training]')
            # import ipdb; ipdb.set_trace()
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            # rgb_gt = rearrange(batch['rgb'], '(h w) c -> c h w', h=h)
            # normal_mono = rearrange(batch['normal_mono'], '(h w) c -> c h w', h=h)
            tensorboard.add_image('img/render', rgb_pred.cpu().numpy(), self.global_step)
            # tensorboard.add_image('img/gt', rgb_gt.cpu().numpy(), self.global_step)
        # if self.global_step<len(self.test_dataset.render_path_rays):
        #     # print('[rendering in training]')
        #     # batch = self.test_dataset[0]
        #     # for i in batch:
        #     #     if isinstance(batch[i], torch.Tensor):
        #     #         batch[i] = batch[i].cuda()
        #     results = self(batch, split='val')
        #     w, h = self.img_wh
        #     rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        #     rgb_frame = (rgb_frame*255).astype(np.uint8)
        #     self.frame_series.append(rgb_frame)
        #     imageio.imsave(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}/', f'test_rgb{self.global_step}.png'), rgb_frame)
        # if self.global_step == len(self.test_dataset.render_path_rays):
        #     imageio.mimsave(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}', "intermediate_circle_path.mp4"),
        #                 self.frame_series,
        #                 fps=30, macro_block_size=1)
        #     self.frame_series = []

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
        self.log('test/psnr', mean_psnr, prog_bar=True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32)

    trainer.fit(system)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

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
    
    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
        
    render_for_test(hparams, split='train')
