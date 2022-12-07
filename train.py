import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
import random
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
# from models.networks import NGP, Normal
from models.networks_sem_4 import NGP, Normal
from models.implicit_mask import implicit_mask
from models.rendering_ import render, MAX_SAMPLES
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


def depth2img(depth):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    depth = depth/16
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def mask2img(mask):
    mask_img = cv2.applyColorMap((mask*255).astype(np.uint8),
                                  cv2.COLORMAP_BONE)

    return mask_img

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
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        if hparams.normal_distillation:
            self.normal_model = Normal()
        if hparams.embed_msk:
            self.msk_model = implicit_mask()
            
        ###
        img_dir_name = None
        if hparams.climate is not None:
            img_dir_name = 'climate/{}'.format(hparams.climate)
        elif os.path.exists(os.path.join(hparams.root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
            img_dir_name = 'rgb'

        
        if hparams.dataset_name == 'kitti':
            self.N_imgs = 2 * hparams.train_frames
        elif hparams.dataset_name == 'mega':
            self.N_imgs = 1920 // 6
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
                  'render_up_sem': hparams.render_normal_up,
                  'render_sem': hparams.render_semantic,
                  'distill_normal': self.global_step//1000>=self.hparams.num_epochs and self.hparams.normal_distillation,
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
                  'use_upLabel': self.hparams.render_normal_up,
                  'depth_mono': self.hparams.depth_mono,
                  'climate': self.hparams.climate}

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

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        if self.hparams.depth_smooth:
            self.register_buffer('directions_xdx', self.train_dataset.directions_xdx.to(self.device))
            self.register_buffer('directions_ydy', self.train_dataset.directions_ydy.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
        
        ckpt_path = None
        if hparams.ckpt_load:
            ckpt_path = hparams.ckpt_load
        load_ckpt(self.model, ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net'])
        print('Loaded checkpoint: {}'.format(ckpt_path))

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]
        
        opts = []
        # self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-8)
        self.net_opt = Adam(net_params, self.hparams.lr, eps=1e-8)
        opts = [self.net_opt]
        if self.hparams.optimize_ext:
            # learning rate is hard-coded
            # pose_r_opt = FusedAdam([self.dR], 1e-6)
            # pose_r_opt = Adam([self.dR], 1e-6)
            # # pose_t_opt = FusedAdam([self.dT], 1e-6)
            # pose_t_opt = Adam([self.dT], 1e-6)
            # opts += [pose_r_opt, pose_t_opt]
            # opts += [Adam([self.dR, self.dT], 1e-6)]
            self.net_opt = Adam([{'params': net_params}, 
                          {'params': [self.dR], 'lr': 1e-8},
                          {'params': [self.dT], 'lr': 1e-8}], lr=self.hparams.lr, eps=1e-8)
            opts = [self.net_opt]
            
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
            # if self.global_step>=self.warmup_steps:
            #     uniform_density = self.model.uniform_sample()

        # with autograd.detect_anomaly():
        results = self(batch, split='train')
        
        if self.hparams.embed_msk:
            # embedding_msk = self.embedding_msk(batch['img_idxs'])
            w, h = self.img_wh
            uv = torch.tensor(batch['uv']).cuda()
            img_idx = torch.tensor(batch['img_idxs']).cuda()
            uvi = torch.zeros((uv.shape[0], 3)).cuda()
            uvi[:, 0] = (uv[:, 0]-h/2) / h
            uvi[:, 1] = (uv[:, 1]-w/2) / w
            uvi[:, 2] = (img_idx - self.N_imgs/2) / self.N_imgs
            mask = self.msk_model(uvi)
        
        # import ipdb; ipdb.set_trace()
        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                    'depth_smooth': self.hparams.depth_smooth if self.global_step//1000>=30 else False,
                    'uniform_density': uniform_density,
                    'up_sem': False,
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
        # self.log('train/grads_inf_cnt', results['gradinf_cnt'])
        # self.log('train/Ro', results['Ro'].mean())
        # self.log('train/Rp', results['Rp'].mean())
        # if self.global_step%5000 == 0:
        #     # for i in range(len(self.hparams.samples)):
        #     xyz_samples = results['xyzs'].reshape(-1, 3).detach()
        #     self.samples_points.append(xyz_samples.cpu().numpy())
        #     self.samples_color_ = np.ones((xyz_samples.shape[0], 4))
        #     self.samples_color_  = self.samples_color_*255
        #     self.samples_color.append(self.samples_color_.astype(np.uint8))
        #     self.samples_points = np.concatenate(self.samples_points, axis=0)
        #     self.samples_color = np.concatenate(self.samples_color, axis=0)
        #     point_cloud = trimesh.points.PointCloud(self.samples_points, self.samples_color)
        #     point_cloud.export(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}/test_samples_{self.global_step}.ply'))
        #     self.samples_points = []
        #     self.samples_color = []
        if self.global_step%5000 == 0 and self.global_step>0 and False:
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
            
            # import ipdb; ipdb.set_trace()
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            # rgb_up_pred = rearrange(results['rgb+up'], '(h w) c -> c h w', h=h)
            semantic_pred = semantic2img(rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h), self.hparams.get('classes', 7))
            semantic_pred  = rearrange(semantic_pred , 'h w c -> c h w', h=h)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            normal_pred = rearrange((results['normal_pred']+1)/2, '(h w) c -> c h w', h=h)
            # normal_raw = rearrange((results['normal_raw']+1)/2, '(h w) c -> c h w', h=h)
            rgb_gt = rearrange(batch['rgb'], '(h w) c -> c h w', h=h)
            # normal_mono = rearrange(batch['normal_mono'], '(h w) c -> c h w', h=h)
            tensorboard.add_image('img/render', rgb_pred.cpu().numpy(), self.global_step)
            # tensorboard.add_image('img/render+up', rgb_up_pred.cpu().numpy(), self.global_step)
            tensorboard.add_image('img/semantic', semantic_pred, self.global_step)
            tensorboard.add_image('img/depth', depth_pred, self.global_step)
            tensorboard.add_image('img/normal_pred', normal_pred.cpu().numpy(), self.global_step)
            # tensorboard.add_image('img/normal_raw', normal_raw.cpu().numpy(), self.global_step)
            tensorboard.add_image('img/gt', rgb_gt.cpu().numpy(), self.global_step)
            

        for name, params in self.model.named_parameters():
            check_nan=None
            check_inf=None
            if params.grad is not None:
                check_nan = torch.any(torch.isnan(params.grad))
                check_inf = torch.any(torch.isinf(params.grad))
            # tensorboard.add_histogram('train/{name}_grad', params.grad, self.global_step)
                # if 'xyz_encoder' in name:
                #     tensorboard.add_histogram(f'train/{name}_grad_max', params.grad, self.global_step)
            # print(f'name {name}, grad_requires {params.requires_grad}, grad_value contains nan? {check_nan}, grad_value contains inf? {check_inf}')
            if check_inf or check_nan:
                import ipdb; ipdb.set_trace()

        return loss
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

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
    if hparams.val_only and (not hparams.ckpt_load):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    if hparams.normal_distillation_only:
        assert hparams.ckpt_load is not None, "No weight ckpt specified when distilling normals"
        hparams.num_epochs = 0
    if not hparams.normal_distillation:
        hparams.normal_epochs = 0
    system = NeRFSystem(hparams)

    if hparams.val_only:
        render_for_test(hparams, split='train')
        quit()

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
    
    trainer = Trainer(max_epochs=hparams.num_epochs+hparams.normal_epochs,
                      check_val_every_n_epoch=hparams.normal_epochs + hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32,
                      gradient_clip_val=50,
                      detect_anomaly=True)

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
