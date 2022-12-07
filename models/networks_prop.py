import torch
from torch import nn
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh
import numpy as np
from einops import rearrange
from .rendering_old import NEAR_DISTANCE

class NGP_prop(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', use_skybox=False, embed_a=False, embed_a_len=12, classes=7):
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
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
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
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        
        if embed_a:
            print("Use embed_a!")
            self.rgb_net = \
                tcnn.Network(
                    n_input_dims=32+embed_a_len, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": self.rgb_act,
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
        else:
            self.rgb_net = \
                tcnn.Network(
                    n_input_dims=32, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": self.rgb_act,
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
                
        self.semantic_net = tcnn.Network(
                    n_input_dims=self.xyz_encoder.n_output_dims, n_output_dims=classes,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
        self.semantic_act = nn.Softmax(dim=-1)
            
        if use_skybox:
            print("Use skybox!")
            self.skybox_dir_encoder = \
                tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 3,
                    },
                )

            self.skybox_rgb_net = \
                tcnn.Network(
                    n_input_dims=9,
                    n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": rgb_act,
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
            
        self.sigma_act = TruncExp.apply

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        # x = (x-self.center+1e-6)/torch.sqrt(torch.abs(x-self.center+1e-6)*self.scale)
        # x = (x+1)/2
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, d, embed_a, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        if self.embed_a:
            rgbs = self.rgb_net(torch.cat([d, h, embed_a], 1))
        else:
            rgbs = self.rgb_net(torch.cat([d, h], 1))
            
        semantic = self.semantic_net(h)
        semantic = self.semantic_act(semantic)
            
        return sigmas, rgbs, semantic

    def forward_skybox(self, d):
        if not self.use_skybox:
            return None
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.skybox_dir_encoder((d+1)/2)
        rgbs = self.skybox_rgb_net(d)
        
        return rgbs     