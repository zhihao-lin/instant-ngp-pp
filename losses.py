import torch
from torch import nn
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
        self.lambda_distortion = 3e-4
        self.lambda_depth_mono = 5e-3
        
        self.Annealing = ExponentialAnnealingWeight(max = 1, min = 6e-2, k = 3e-4)
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, results, target, **kwargs):
        d = {}
        if kwargs.get('distill_normal', False):
            # valid_mask = torch.logical_and(results['depth']>1e-9, results['depth']<4).unsqueeze(-1)
            d['normal_sim'] = (results['normal_pred']-results['normal_raw'])**2
            d['normal_smooth'] = 0.1*(results['normal_pred']-results['normal_pred_e'])**2
            # print('[LOSS] normal_sim contains nan?', torch.any(torch.isnan(d['normal_sim'])))
            # print('[LOSS] normal_sim contains inf?', torch.any(torch.isinf(d['normal_sim'])))
            # print('[LOSS] normal_smooth contains nan?', torch.any(torch.isnan(d['normal_smooth'])))
            # print('[LOSS] normal_smooth contains inf?', torch.any(torch.isinf(d['normal_smooth'])))
        else:
            if kwargs.get('distill', False):
                labels = torch.argmax(results['semantic'].detach(), dim=-1)
                mask = torch.logical_and(labels==0, labels==3)
                # mask = torch.logical_and(labels==6, mask)
                # mask = ~mask
                weight = torch.zeros_like(mask, dtype=torch.float32).cuda()
                # prob_raw, _ = nn.functional.softmax(results['semantic'].detach(), dim=-1).max(-1)
                weight[mask] = 1
                weight[~mask] = 1e-4
                # d['rgb'] = 0.1 * weight.unsqueeze(-1) * (kwargs['iso_color']-target['rgb'])**2 # let iso color learn the ground color
                d['rgb'] = 0.1 * weight.unsqueeze(-1) * (results['rgb']-target['rgb'])**2 # let metaballs learn the ground color
                d['rgb_0'] = (results['rgb_0']-target['rgb'])**2
                # d['snow_reg'] = (results['rgbs_mb'] - kwargs['iso_color'].detach())**2
            else: 
                if kwargs.get('embed_msk', False):
                    d['r_ms'], _ = self.mask_regularize(kwargs['mask'], self.Annealing.getWeight(kwargs['step']), 0)
                    d['rgb'] = (1-kwargs['mask']) * (results['rgb']-target['rgb'])**2
                else:
                    d['rgb'] = (results['rgb']-target['rgb'])**2
            
                o = results['opacity']+1e-10
                # encourage opacity to be either 0 or 1 to avoid floater
                d['opacity'] = self.lambda_opa*(-o*torch.log(o))
            
                # if kwargs['dataset_name']=='tnt':
                # d['sparsity'] = .0001*(1.0-torch.exp(-0.01*results['sigma']).mean())
                # if kwargs['uniform_density'] is not None:
                #     d['sparsity_uniform'] = .001*(1.0-torch.exp(-0.01*kwargs['uniform_density']).mean())
                # depth smooth loss
                # import ipdb; ipdb.set_trace()
                # if kwargs['depth_smooth']:
                #     d['depth1'] = 0.00001*(results['depth']-results['depth1'])**2
                #     d['depth2'] = 0.00001*(results['depth']-results['depth2'])**2
                # gradient_d = (results['gradient_d_o']-torch.bmm((results['gradient_d_o']).unsqueeze(1),results['rays_d'].unsqueeze(-1)).squeeze(-1)*results['rays_d'])**2
                # d['depth_reg'] = 1e-6*torch.clamp(torch.sum(gradient_d), min=-20.0, max=20.0)
                # d['Ro'] = (results['Ro']-torch.zeros_like(results['Ro']).cuda()) * 5e-3
                if kwargs.get('normal_p', False):
                    # d['Rp'] = (results['Rp']-torch.zeros_like(results['Rp']).cuda()) * 2e-4 # for non-ref-nerf model
                    d['Rp'] = (results['Rp']-torch.zeros_like(results['Rp']).cuda()) * 7e-4 # for ref-nerf model
                # d['norm_l2'] = (results['normal_pred']-results['normal_raw'])**2
                
                if self.lambda_distortion > 0:
                    d['distortion'] = self.lambda_distortion * \
                    DistortionLoss.apply(results['ws'], results['deltas'],
                                            results['ts'], results['rays_a'])

                # if kwargs.get('up_sem', False):
                #     d['up_sem'] = -4e-2*(target['normal_up']*torch.log(torch.clamp(results['up_sem'], min=1e-6, max=1.))+(1-target['normal_up'])*torch.log(torch.clamp(1-results['up_sem'], min=1e-6, max=1.)))
                
                if kwargs.get('semantic', False):
                    d['CELoss'] = 4e-2*self.CrossEntropyLoss(results['semantic'], target['label'])
                    sky_mask = torch.where(target['label']==4, 1., 0.)
                    d['sky_depth'] = 1e-1*sky_mask*torch.exp(-results['depth'])
                    # if d['CELoss']<0:
                    #     import ipdb; ipdb.set_trace()
                if kwargs.get('depth_mono', False):
                    # import ipdb; ipdb.set_trace()
                    depth_2d = target['depth'] * 10 + 0.5
                    scale, shift = compute_scale_and_shift(results['depth'].detach(), depth_2d)
                    d['depth_mono'] = self.lambda_depth_mono * torch.exp(-target['depth']) * (scale * results['depth'] + shift - depth_2d)**2
                    
            flag = 0
            for (i, n) in d.items():
                if torch.any(torch.isnan(n)):
                    print(f'nan in d[{i}]')
                    print(f'max: {torch.max(n)}')
                    flag = 1
            
            # assert flag == 0, 'nan occurs!'
            
        return d

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit