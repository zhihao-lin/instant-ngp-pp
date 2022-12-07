from venv import create
from warnings import catch_warnings
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh, ReLU
import numpy as np
from einops import rearrange
from .rendering_old import NEAR_DISTANCE
from .ref_util import *

class Normal(nn.Module):
    def __init__(self, width=128, depth=5):
        super().__init__()
        
        self.normal_net = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3, n_output_dims=3,
                    encoding_config={
                        "otype": "Frequency",
                        "n_frequencies": 6
                    },
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": width,
                        "n_hidden_layers": depth,
                    }
                )
    def forward(self, x, **kwargs):
        out = self.normal_net(x)
        return out
    
class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', use_skybox=False, embed_a=False, embed_a_len=12):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.use_skybox = use_skybox
        self.embed_a = embed_a
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 32; F = 2; log2_T = 21; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        bottle_neck = 16
        
        # self.xyz_encoder = \
        #     tcnn.NetworkWithInputEncoding(
        #         n_input_dims=3, n_output_dims=bottle_neck + 11,
        #         encoding_config={
        #             "otype": "Grid",
	    #             "type": "Hash",
        #             "n_levels": L,
        #             "n_features_per_level": F,
        #             "log2_hashmap_size": log2_T,
        #             "base_resolution": N_min,
        #             "per_level_scale": b,
        #             "interpolation": "Linear"
        #         },
        #         network_config={
        #             "otype": "CutlassMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 1,
        #         }
        #     )
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
            )
            
        self.xyz_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True)
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(64, 1)
        )
        
        self.feat_net = nn.Sequential(
            nn.Linear(64, bottle_neck+3)
        )
        
        # self.dir_encoder = \
        #     tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "Frequency",
        #             "n_frequencies": 4
        #         },
        #     )
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        dir_encoder_output_dim = self.dir_encoder.n_output_dims
        print(f'dir_enc_dim={dir_encoder_output_dim}')
        
        # self.dir_encoder = generate_ide_fn(5)
        # dir_encoder_output_dim = 72
        
        if embed_a:
            print("Use embed_a!")
            self.rgb_net1 = \
                tcnn.Network(
                    n_input_dims=bottle_neck+dir_encoder_output_dim+1+embed_a_len, n_output_dims=64,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "ReLU",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
            self.rgb_net2 = \
                tcnn.Network(
                    n_input_dims=64+bottle_neck+dir_encoder_output_dim+1+embed_a_len, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
        else:
            self.rgb_net1 = \
                tcnn.Network(
                    n_input_dims=bottle_neck+dir_encoder_output_dim+1, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 128,
                        "n_hidden_layers": 3,
                    }
                )
            # self.rgb_net2 = \
            #     tcnn.Network(
            #         n_input_dims=64+bottle_neck+dir_encoder_output_dim+1, n_output_dims=3,
            #         network_config={
            #             "otype": "CutlassMLP",
            #             "activation": "ReLU",
            #             "output_activation": "None",
            #             "n_neurons": 64,
            #             "n_hidden_layers": 3,
            #         }
            #     )
            
        if use_skybox:
            print("Use skybox!")
            self.skybox_dir_encoder = \
                tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                )

            self.skybox_rgb_net = \
                tcnn.Network(
                    n_input_dims=16,
                    n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": rgb_act,
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
            
        # self.sigma_act = TruncExp.apply
        self.sigma_act = nn.Softplus()
        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "CutlassMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x)
        h = self.xyz_net(h)
        sigmas = self.sigma_net(h)
        # sigmas = TruncExp.apply(h[:, 0])
        # import ipdb; ipdb.set_trace()
        sigmas = self.sigma_act(sigmas[:, 0])
        if return_feat: 
            feat = self.feat_net(h)
            return sigmas, feat
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs
    
    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        x = x.requires_grad_(True)
        try:
            sigmas, h = self.density(x, return_feat=True)
        except:
            import ipdb; ipdb.set_trace()
        normals=None
        try:
            grads = torch.autograd.grad(
                outputs=sigmas,
                inputs=x,
                grad_outputs=torch.ones_like(sigmas, requires_grad=False).cuda(),
                retain_graph=True
                )[0]
        except:
            import ipdb; ipdb.set_trace()
        # grads = grads.detach()
        # import ipdb; ipdb.set_trace()
        # eps = 1e-6
        # x_dx = x
        # x_dx[:, 0] = x_dx[:, 0]+eps
        # sigmas_dx = self.density(x_dx)
        # x_dy = x
        # x_dy[:, 1] = x_dx[:, 1]+eps
        # sigmas_dy = self.density(x_dy)
        # x_dz = x
        # x_dz[:, 2] = x_dz[:, 2]+eps
        # sigmas_dz = self.density(x_dz)
        # grads = torch.zeros_like(x)
        # grads[:, 0] = (sigmas_dx-sigmas)/eps
        # grads[:, 1] = (sigmas_dy-sigmas)/eps
        # grads[:, 2] = (sigmas_dz-sigmas)/eps
        
        # if torch.any(torch.isnan(grads)):
        #     print('grads contains nan')
        # if torch.any(torch.isinf(grads)):
        #     print('grads contains inf')
        # grads_.detach()
        mask = torch.logical_or(torch.isinf(grads).any(-1), torch.isnan(grads).any(-1))
        mask = ~mask
        # x_ = x[mask]
        # x_.requires_grad_(True)
        # sigmas_new = self.density(x_)
        # grads = torch.autograd.grad(
        #     outputs=sigmas_new,
        #     inputs=x_,
        #     grad_outputs=torch.ones_like(sigmas_new, requires_grad=False),
        #     retain_graph=True,
        #     create_graph=True
        #     )[0]
                
        # normals = torch.ones_like(x).cuda()*1e-6
        # normals[mask] = grads[mask]
        # normals = grads
        grads.nan_to_num(1e-6)
        normals = -F.normalize(grads, p=2, dim=-1, eps=1e-6)
        # if normals.isnan().any():
        #     import ipdb; ipdb.set_trace()
        
        # import ipdb; ipdb.set_trace()
        bottle_neck = h[:, 3:]
        # cd = h[:, 1:4]
        # s = F.sigmoid(h[:, 4:7])
        # roughness = F.softplus(h[:, 7:8]-1)
        normals_pred_ = h[:, 0:3]
        # normals_pred = torch.ones_like(normals_pred_).cuda()*1e-6
        # normals_pred[mask] = normals_pred_[mask]
        normals_pred = -F.normalize(normals_pred_, p=2, dim=-1, eps=1e-6)
        
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        refdir = reflect(-d, normals_pred)
        # d_enc = self.dir_encoder(refdir, roughness)
        d_enc = self.dir_encoder(refdir)
        dotprod = torch.sum(-normals_pred*d, dim=-1, keepdim=True)
        
        if self.embed_a:
            rgbs = self.rgb_net1(torch.cat([d_enc, bottle_neck, dotprod, kwargs['embedding_a']], 1))
            rgbs = self.rgb_net2(torch.cat([rgbs, d_enc, bottle_neck, dotprod, kwargs['embedding_a']], 1))
        else:
            rgbs = self.rgb_net1(torch.cat([d_enc, bottle_neck, dotprod], 1))
            # rgbs = self.rgb_net2(torch.cat([rgbs, d_enc, bottle_neck, dotprod], 1))

        rgbs = F.sigmoid(rgbs)

        # diffuse_color = F.sigmoid(cd-torch.log(3.0*torch.ones_like(cd)))
        # rgbs = torch.clamp(diffuse_color+s*rgbs, 0.0, 1.0)                
            
        return sigmas, rgbs, normals, normals_pred
    
    def forward_test(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        try:
            sigmas, h = self.density(x, return_feat=True)
        except:
            import ipdb; ipdb.set_trace()
        bottle_neck = h[:, 3:]
        normals_pred_ = h[:, 0:3]
        normals_pred = -F.normalize(normals_pred_, p=2, dim=-1, eps=1e-6)
        
        
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        refdir = reflect(-d, normals_pred)
        # d_enc = self.dir_encoder(refdir, roughness)
        d_enc = self.dir_encoder(refdir)
        dotprod = torch.sum(-normals_pred*d, dim=-1, keepdim=True)
        
        if self.embed_a:
            rgbs = self.rgb_net1(torch.cat([d_enc, bottle_neck, dotprod, kwargs['embedding_a']], 1))
            rgbs = self.rgb_net2(torch.cat([rgbs, d_enc, bottle_neck, dotprod, kwargs['embedding_a']], 1))
        else:
            rgbs = self.rgb_net1(torch.cat([d_enc, bottle_neck, dotprod], 1))
            rgbs = self.rgb_net2(torch.cat([rgbs, d_enc, bottle_neck, dotprod], 1))

        rgbs = F.sigmoid(rgbs)              
            
        return sigmas, rgbs, normals_pred

    def forward_skybox(self, d):
        if not self.use_skybox:
            return None
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.skybox_dir_encoder((d+1)/2)
        rgbs = self.skybox_rgb_net(d)
        
        return rgbs     

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3) batch transpose
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
            # density_threshold = -1.
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)
        
        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
    
    def uniform_sample(self, resolution=128):
        half_grid_size = self.scale / resolution
        samples = torch.stack(torch.meshgrid(
                torch.linspace(0, 1-half_grid_size, resolution),
                torch.linspace(0, 1-half_grid_size, resolution),
                torch.linspace(0, 1-half_grid_size, resolution),
            ), -1).cuda()
        dense_xyz = self.xyz_min * (1-samples) + self.xyz_max * samples
        dense_xyz += half_grid_size*torch.rand_like(dense_xyz).cuda()
        density = self.density(dense_xyz.view(-1,3))
        return density