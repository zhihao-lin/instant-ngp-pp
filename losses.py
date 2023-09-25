import torch
from torch import nn
import torch.nn.functional as F
import vren
import math

def compute_scale_and_shift(prediction, target):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(prediction * prediction)
    a_01 = torch.sum(prediction)
    ones = torch.ones_like(prediction)
    a_11 = torch.sum(ones)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(prediction * target)
    b_1 = torch.sum(target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    # x_0 = torch.zeros_like(b_0)
    # x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0 = torch.FloatTensor(0).cuda()
        x_1 = torch.FloatTensor(0).cuda()

    return x_0, x_1

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)
    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None
    
class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))

class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_opa = 2e-4
        self.lambda_distortion = 3e-4 # default
        # self.lambda_distortion = 1e-4 # for meganerf
        self.lambda_depth_mono = 1
        self.lambda_normal_mono = 1e-3
        self.lambda_normal_ref_rp = 1e-3
        self.lambda_normal_ref_ro = 1e-3
        self.lambda_sky = 1e-1
        self.lambda_semantic = 4e-2
        
        self.Annealing = ExponentialAnnealingWeight(max = 1, min = 6e-2, k = 1e-3)
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=256)

    def forward(self, results, target, **kwargs):
        d = {}
        
        if kwargs.get('embed_msk', False):
            d['r_ms'], _ = self.mask_regularize(kwargs['mask'], self.Annealing.getWeight(kwargs['step']), 0)
            d['rgb'] = (1-kwargs['mask']) * (results['rgb']-target['rgb'])**2
        else:
            d['rgb'] = (results['rgb']-target['rgb'])**2
    
        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opa*(-o*torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
            DistortionLoss.apply(results['ws'], results['deltas'],
                                    results['ts'], results['rays_a'])
        
        if kwargs.get('normal_ref', False):# for ref-nerf model
            d['normal_ref_rp'] = self.lambda_normal_ref_rp * results['Rp'] 
            d['normal_ref_ro'] = self.lambda_normal_ref_ro * results['Ro'] 
        
        if kwargs.get('normal_mono', False):
            # d['normal_mono'] = self.lambda_normal_mono * torch.exp(-results['depth'].detach()/kwargs.get('scale', 1))[:, None] * (target['normal']-results['normal_pred'])**2
            # d['normal_mono'] = (target['normal']-results['normal_pred'])**2
            normal_pred = F.normalize(results['normal_pred'], dim=-1)
            normal_gt   = F.normalize(target['normal'], dim=-1)
            l1_loss = torch.abs(normal_pred - normal_gt) #(n, 3)
            cos_loss = -(normal_pred * normal_gt) #(n, 3)
            d['normal_mono'] = self.lambda_normal_mono * (l1_loss + 0.1 * cos_loss)

        if kwargs.get('semantic', False):
            d['CELoss'] = self.lambda_semantic*self.CrossEntropyLoss(results['semantic'], target['label'])
            sky_mask = torch.where(target['label']==4, 1., 0.)
            d['sky_depth'] = self.lambda_sky*sky_mask*torch.exp(-results['depth'])

        if kwargs.get('depth_mono', False): # for kitti360 dataset
            depth_2d = target['depth'] / 25
            mask = depth_2d>0
            weight = torch.zeros_like(depth_2d).cuda()
            weight[mask] = 1.
            scale, shift = compute_scale_and_shift(results['depth'][mask].detach(), depth_2d[mask])
            d['depth_mono'] = weight * self.lambda_depth_mono * torch.exp(-results['depth'].detach()/kwargs.get('scale', 1)) * (scale * results['depth'] + shift - depth_2d)**2
                
        flag = 0
        for (i, n) in d.items():
            if torch.any(torch.isnan(n)):
                print(f'nan in d[{i}]')
                print(f'max: {torch.max(n)}')
                flag = 1
            
        return d

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit