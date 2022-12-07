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
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from datasets.snow import SnowSeed

# models
from kornia.utils.grid import create_meshgrid3d
# from models.networks import NGP, Normal
from models.networks_metaball import NGP
from models.rendering_metaball import render, MAX_SAMPLES, metaball_seeds
from models.global_var import global_var

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses_metaball import MetaballLoss

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
        self.automatic_optimization = False
        self.save_hyperparameters(hparams)
        
        self.warmup_steps = 128
        self.to_zero = 0
        self.update_interval = 16
        self.step_ = 0

        self.loss = MetaballLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False
                        
        rgb_act = 'Sigmoid'
        self.normal_model = None
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)

        dict = torch.load(hparams.ckpt_path)
        self.up = dict['up'].cuda()
        self.ground_height = dict['ground_height'].cuda()
        
        ###
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
                # if hparams.ckpt_path is not None:
                #     load_ckpt(self.embedding_a, hparams.ckpt_path, model_name='embedding_a', \
                #     prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])   
        ###
        
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        self.samples_points = []
        self.samples_color = []
        self.cam_o = []
        self.cam_color = []
        self.center_o = []
        self.center_color = []

    def forward(self, batch, split, **kwargs):
        # import ipdb; ipdb.set_trace()
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
            embedding_a = self.embedding_a(torch.tensor([0]).cuda())

        rays_o, rays_d = get_rays(directions, poses_)

        kwargs_ = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox if self.step_>=self.warmup_steps else False,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_up_sem': hparams.render_normal_up,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'up_vector': self.up,
                  'centers': self.center_o
                  }
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
            kwargs_['exp_step_factor'] = 1/256
        if self.hparams.embed_a:
            kwargs_['embedding_a'] = embedding_a
        if 'samples' in kwargs.keys():
            kwargs_['samples'] = kwargs['samples']

        if split == 'train':
            return render(self.model, rays_o, rays_d, **kwargs_)
        else:
            chunk_size = 8192*16
            all_ret = {}
            for i in range(0, rays_o.shape[0], chunk_size):
                ret = render(self.model, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **kwargs_)
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
        
        self.snow_seed = SnowSeed(self.up)     

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
        
        load_ckpt(self.model, self.hparams.ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net'])
        load_ckpt(self.embedding_a, self.hparams.ckpt_path, model_name='embedding_a', prefixes_to_ignore=['model', 'normal_net'])

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]
        
        opts = []
        # self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-8)
        self.net_opt = Adam(net_params, lr=1e-2, eps=1e-15)
        opts = [self.net_opt]
        if self.hparams.optimize_ext:
            self.net_opt = Adam([{'params': net_params}, 
                          {'params': [self.dR], 'lr': 1e-8},
                          {'params': [self.dT], 'lr': 1e-8}], lr=self.hparams.lr, eps=1e-8)
            opts = [self.net_opt]
            
        # net_sch = CosineAnnealingLR(self.net_opt,
        #                             self.hparams.num_epochs+self.hparams.normal_epochs,
        #                             self.hparams.lr/30)       

        return opts

    def train_dataloader(self):
        return DataLoader(self.snow_seed,
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
        opt = self.optimizers(use_pl_optimizer=True)
        
        if self.step_>=self.to_zero:
            uniform_density = None
            # if self.step_%self.update_interval == 0:
            #     self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
            #                                 warmup=self.step_<self.to_zero+self.warmup_steps)
            
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            rays_o, rays_d = get_rays(directions, poses)
            kwargs = {'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_up_sem': hparams.render_normal_up,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'up_vector': self.train_dataset.up
                  }
            if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
                kwargs['exp_step_factor'] = 1/256
            if self.hparams.embed_a:
                kwargs['embedding_a'] = self.embedding_a(batch['img_idxs'])
            density_r_target, samples, depth_proxy, centers, samples_out = metaball_seeds(self.model, rays_o, rays_d, **kwargs)
            # self.samples_points.append(samples.cpu().numpy())
            # # self.samples_color_ = density_r_target.unsqueeze(-1).repeat(1,4)/60
            # # self.samples_color_[:, 3] = 1.
            # # self.samples_color_ = self.samples_color_.cpu().numpy()
            # self.samples_color_ = np.ones((density_r_target.shape[0], 4))
            # self.samples_color_  = self.samples_color_*255
            # self.samples_color.append(self.samples_color_.astype(np.uint8))
            
            # self.cam_o.append(rays_o.cpu().numpy())
            # self.cam_color_ = np.zeros((rays_o.shape[0], 4))
            # self.cam_color_[:, 0] = 1.
            # self.cam_color_[:, 3] = 1.
            # self.cam_color.append((self.cam_color_*255).astype(np.uint8))
            
            self.center_o.append(centers.cpu().numpy())
            self.center_color_ = np.zeros((centers.shape[0], 4))
            self.center_color_[:, 1] = 1.
            self.center_color_[:, 3] = 1.
            self.center_color.append((self.center_color_*255).astype(np.uint8))
            
            # for _ in range(100):    
            #     kwargs['samples'] = samples
            #     density_r_pred, _ = self.model.forward_mb(samples)
            #     density_r_pred_out, _ = self.model.forward_mb(samples_out)
            #     results = self(batch, split='train', **kwargs)
                
            #     loss_kwargs = {'density_r_target': density_r_target,
            #                    'depth_proxy': depth_proxy}
            #     loss_d = self.loss(results, batch, **loss_kwargs)
            #     loss = sum(lo.mean() for lo in loss_d.values())
            #     # loss = torch.mean((density_r_target-density_r_pred)**2)+1e-2*torch.mean((results['depth']-depth_proxy)**2)
            #     loss = torch.mean((density_r_target-density_r_pred)**2)+torch.mean((density_r_pred_out)**2)
            #     # loss = torch.mean(())
                
            #     opt.zero_grad()
            #     self.manual_backward(loss)
            #     opt.step()
            # self.log('lr', self.net_opt.param_groups[0]['lr'])
            # self.log('train/loss', loss, True)
                
            # uniform_density = self.model.uniform_mb_sample()
            # loss = torch.mean(0.001*(1.0-torch.exp(-0.001*uniform_density).mean()))
            # opt.zero_grad()
            # self.manual_backward(loss)
            # opt.step()
            
            # import ipdb; ipdb.set_trace()
            # if torch.isnan(loss):
            #     import ipdb; ipdb.set_trace()
            # self.log('train/s_per_ray', results['total_samples']/len(batch['rgb']), True)
            if self.step_%50 == 0 and self.step_>0:
                print('[val in training]')
                
                # self.samples_points = np.concatenate(self.samples_points, axis=0)
                # self.samples_color = np.concatenate(self.samples_color, axis=0)
                # self.cam_o = np.concatenate(self.cam_o, axis=0)
                # self.cam_color = np.concatenate(self.cam_color, axis=0)
                self.center_o = np.concatenate(self.center_o, axis=0)
                self.center_color = np.concatenate(self.center_color, axis=0)
                # selected_num = int(len(self.center_o)*0.9)
                # indices = torch.randperm(len(self.center_o))[:selected_num]
                # self.center_o = self.center_o[indices]
                # self.center_color = self.center_color[indices]
                print(f'metaball_num:{self.center_o.shape[0]}')
                # pts = np.concatenate([self.samples_points, self.cam_o, self.center_o], axis=0)
                # colors = np.concatenate([self.samples_color, self.cam_color, self.center_color], axis=0)
                # pts = np.concatenate([self.cam_o, self.center_o], axis=0)
                # colors = np.concatenate([self.cam_color, self.center_color], axis=0)
                # point_cloud = trimesh.points.PointCloud(pts, colors)
                point_cloud = trimesh.points.PointCloud(self.center_o, self.center_color)
                point_cloud.export(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}/test_samples_{self.step_}.ply'))
                # import ipdb; ipdb.set_trace()
                # batch = self.test_dataset[random.randint(0, math.ceil(self.N_imgs/8)-1)]
                
                for img_idx in range(len(self.test_dataset)):
                    print(img_idx)
                    batch = self.test_dataset[img_idx+1]
                    for i in batch:
                        if isinstance(batch[i], torch.Tensor):
                            batch[i] = batch[i].cuda()
                    results = self(batch, split='test')
                    w, h = self.img_wh
                    rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                    rgb_frame = (rgb_frame*255).astype(np.uint8)
                    imageio.imsave(os.path.join(f'logs/{hparams.dataset_name}/{hparams.exp_name}', f'test_rgb{img_idx}.png'), rgb_frame)
                
                batch = self.test_dataset[0]
                for i in batch:
                    if isinstance(batch[i], torch.Tensor):
                        batch[i] = batch[i].cuda()
                results = self(batch, split='test')
                w, h = self.img_wh
                rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
                # rgb_up_pred = rearrange(results['rgb+up'], '(h w) c -> c h w', h=h)
                depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
                rgb_gt = rearrange(batch['rgb'], '(h w) c -> c h w', h=h)
                # normal_mono = rearrange(batch['normal_mono'], '(h w) c -> c h w', h=h)
                tensorboard.add_image('img/render', rgb_pred.cpu().numpy(), self.step_)
                # tensorboard.add_image('img/render+up', rgb_up_pred.cpu().numpy(), self.step_)
                tensorboard.add_image('img/depth', depth_pred, self.step_)
                tensorboard.add_image('img/gt', rgb_gt.cpu().numpy(), self.step_)
                
                self.samples_points = []
                self.samples_color = []
                self.cam_o = []
                self.cam_color = []
                self.center_o = []
                self.center_color = []
            
            for name, params in self.model.named_parameters():
                check_nan=None
                check_inf=None
                if params.grad is not None:
                    check_nan = torch.any(torch.isnan(params.grad))
                    check_inf = torch.any(torch.isinf(params.grad))
                # tensorboard.add_histogram('train/{name}_grad', params.grad, self.step_)
                    # if 'xyz_encoder' in name:
                    #     tensorboard.add_histogram(f'train/{name}_grad_max', params.grad, self.step_)
                # print(f'name {name}, grad_requires {params.requires_grad}, grad_value contains nan? {check_nan}, grad_value contains inf? {check_inf}')
                if check_inf or check_nan:
                    import ipdb; ipdb.set_trace()
            
            self.step_ += 1
            # return loss
        else:
            uniform_density = self.model.uniform_mb_sample()
                    
            loss = torch.mean(uniform_density-0)

            self.log('lr', self.net_opt.param_groups[0]['lr'])
            self.log('train/loss for density_zero', loss, True)
            
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            
            self.step_ += 1
            # return loss
        
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
        self.log('test/psnr', mean_psnr, True)

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
