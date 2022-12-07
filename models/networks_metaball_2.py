from venv import create
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh
import numpy as np
from einops import rearrange
from .rendering_metaball_1 import NEAR_DISTANCE, center_density, dkernel_function, remapping_2d, remapping_3d, remapping_height, kernel_function, wrap_light
from .ref_util import *
    
class NGP_mb(nn.Module):
    def __init__(self, scale, up, ground_height, R, R_inv, interval, b=2, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        self.up = up
        self.ground_height = ground_height
        self.interval = interval
        self.R = R
        self.R_inv = R_inv
        self.mb_cascade = 5
        self.b = b

        # constants
        L_mb = 16; F_mb = 2; log2_T_mb = 19; N_min_mb = 16
        b_mb = np.exp(np.log(2048*scale/N_min_mb)/(L_mb-1))
        print(f'GridEncoding for metaball: Nmin={N_min_mb} b={b_mb:.5f} F={F_mb} L={L_mb}')

        self.mb_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L_mb,
                    "n_features_per_level": F_mb,
                    "log2_hashmap_size": log2_T_mb,
                    "base_resolution": N_min_mb,
                    "per_level_scale": b_mb,
                    "interpolation": "Linear"
                })
        
        self.mb_net = nn.Sequential(
            nn.Linear(self.mb_encoder.n_output_dims, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

        self.mb_act = nn.Sigmoid()
        self.iso_color = nn.Parameter(torch.ones(3).cuda())
        
        # # constants
        # L_ = 16; F_ = 2; log2_T_ = 19; N_min_ = 16
        # b_ = np.exp(np.log(2048*scale/N_min_)/(L_-1))
        # print(f'GridEncoding for RGB: Nmin={N_min_} b={b_:.5f} F={F_} T=2^{log2_T_} L={L_}')

        # self.rgb_encoder = \
        #     tcnn.Encoding(3, {
        #         "otype": "HashGrid",
        #         "n_levels": L_,
        #         "n_features_per_level": F_,
        #         "log2_hashmap_size": log2_T_,
        #         "base_resolution": N_min_,
        #         "per_level_scale": b_,
        #         "interpolation": "Linear"
        #     })

        # self.dir_encoder = \
        #     tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "SphericalHarmonics",
        #             "degree": 4,
        #         },
        #     )
        
        # rgb_input_dim = self.rgb_encoder.n_output_dims + self.dir_encoder.n_output_dims
        # print(f'rgb_input_dim: {rgb_input_dim}')
        # self.rgb_net = \
        #     tcnn.Network(
        #         n_input_dims=rgb_input_dim,
        #         n_output_dims=3,
        #         network_config={
        #             "otype": "CutlassMLP",
        #             "activation": "ReLU",
        #             "output_activation": rgb_act,
        #             "n_neurons": 64,
        #             "n_hidden_layers": 1,
        #         }
        #     )

    def alpha(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """

        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.mb_encoder(x)
        h = self.mb_net(h)
        alphas = self.mb_act(h[:, 0])
        
        return alphas
    
    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3)
        alphas = self.alpha(x)
        # rgbs = torch.ones((alphas.shape[0], 3)).cuda()
        
        # feat_rgb = self.rgb_encoder(x)
        
        # d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        # d = self.dir_encoder((d+1)/2)
        
        rgbs = self.iso_color
        
        return alphas, rgbs

    def forward_test(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        with torch.set_grad_enabled(not kwargs.get('stylize', False)):
            # convert x in colmap coordinate into snow falling coordinate
            x_sf = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3)
            # x_sf = x
            N_samples = x_sf.shape[0]
            
            x_sf_vertices = []
            x_sf_dis = []
            x_sf_radius = []
            x_sf_grad = []
            
            for i in range(self.mb_cascade):
                radius = self.interval / self.b**i
                x_sf_coord = torch.floor(x_sf/radius) * radius
                offsets = radius * torch.FloatTensor([[0, 0, 0],
                                                    [0, 0, 1],
                                                    [0, 1, 0],
                                                    [1, 0, 0],
                                                    [0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0],
                                                    [1, 1, 1]]).cuda()
                # offsets_delta = self.interval / (self.b**self.mb_cascade) * (2*torch.rand(offsets.shape).cuda()-1)
                # offsets = offsets + offsets_delta
                x_sf_vertices_ = (x_sf_coord[:, None, :] + offsets[None, :, :]) # (N_samples, 8, 3)
                x_sf_dis_ = torch.norm(x_sf_vertices_-x_sf[:, None, :], dim=-1) # (N_samples, 8)
                x_sf_grad_ = (x_sf[:, None, :] - x_sf_vertices_) / (x_sf_dis_[..., None]) # (N_samples, 8, 3)
                x_sf_vertices.append(x_sf_vertices_.reshape(-1, 3))
                x_sf_dis.append(x_sf_dis_.reshape(-1))
                x_sf_radius.append(radius * torch.ones((x_sf_vertices_.shape[0] * x_sf_vertices_.shape[1])).cuda())
                x_sf_grad.append(x_sf_grad_.reshape(-1, 3))
            
            x_sf_vertices = torch.cat(x_sf_vertices, dim=0) # (N_samples * mb_cascade * 8, 3)
            x_sf_dis = torch.cat(x_sf_dis, dim=0) # (N_samples * mb_cascade * 8)
            x_sf_radius = torch.cat(x_sf_radius, dim=0) # (N_samples * mb_cascade * 8)
            x_sf_grad = torch.cat(x_sf_grad, dim=0) # (N_samples * mb_cascade * 8, 3)
            
            alpha = self.alpha(x_sf_vertices) # (N_samples * mb_cascade * 8)
            # this can be further finetuned
            density_c = alpha * center_density # (N_samples * mb_cascade * 8)
            density_sample = kernel_function(density_c, x_sf_radius, x_sf_dis) # (N_samples * mb_cascade * 8)
            ddensity_sample_dxsf = dkernel_function(density_c, x_sf_radius, x_sf_dis)[:, None] * x_sf_grad
            
            densities = torch.chunk(density_sample, self.mb_cascade, dim=0)
            sigmas = torch.stack(densities, dim=-1) # (N_samples * 8, mb_cascade)
                    
            # rgbs = torch.ones_like(x_sf_dir).cuda() * wrap_light(self.up, x_sf_dir)[:, None] # (N_samples * mb_cascade * 8, 3)
            # rgbs = torch.chunk(rgbs, self.mb_cascade, dim=0)
            # rgbs = torch.stack(rgbs, dim=1).reshape(N_samples, 8*self.mb_cascade, 3) * sigmas.view(N_samples, 8*self.mb_cascade, 1) # (N_samples, 8*mb_cascade, 3)
            # rgbs = rgbs / torch.sum(sigmas.view(N_samples, 8*self.mb_cascade), dim=-1).view(N_samples, 1, 1)
            sigmas = torch.sum(sigmas.view(N_samples, self.mb_cascade*8), dim=-1)
            
            ddensities_dxsf = torch.chunk(ddensity_sample_dxsf, self.mb_cascade, dim=0)
            normals = torch.stack(ddensities_dxsf, dim=1) # (N_samples*8, mb_cascade, 3)
            normals = torch.sum(normals.reshape(N_samples, 8*self.mb_cascade, 3), dim=1) # (N_samples, 3)
            normals = -F.normalize(normals, dim=-1)
        # rgbs = torch.ones((sigmas.shape[0], 3)).cuda()
        # feat_rgb = self.rgb_encoder(x)
        # d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        # d = self.dir_encoder((d+1)/2)
        # rgbs = self.rgb_net(torch.cat([d, feat_rgb], 1)) * wrap_light(torch.FloatTensor([0, -1., 0]).cuda(), normals)[:, None]
        # rgbs = self.iso_color * wrap_light(torch.FloatTensor([0, -1., 0]).cuda(), normals)[:, None]
        rgbs = self.iso_color
        # rgbs = torch.clamp(torch.sum(rgbs, dim=1), max=1.)
        
        return sigmas, rgbs
