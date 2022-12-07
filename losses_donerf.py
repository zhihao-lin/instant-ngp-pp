import torch
from torch import nn
import vren

def distortion_loss(weights, z_samples):
    # weights = weights.squeeze(0)
    # z_samples = z_samples.squeeze(0)
    n_rays, n_samples = weights.shape[-2:]
    z_mids = .5 * (z_samples[..., 1:] + z_samples[..., :-1])
    z_intervals = torch.cat([z_samples[..., :1], z_mids, z_samples[..., -1:]], -1)
    z_mids = .5 * (z_intervals[..., 1:] + z_intervals[..., :-1])
    weights_mat = (weights * z_mids)[..., None, :] * weights.reshape([n_rays, n_samples, 1])
    weights_mat = .5 * torch.abs(weights_mat - weights_mat.transpose(-1,-2))
    distortion_loss = torch.mean(torch.sum(weights_mat, (-2,-1)) + torch.sum(1/3 * (weights*weights*(z_intervals[..., 1:]-z_intervals[..., :-1])), -1))

    return distortion_loss

class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_opa = 1e-3
        self.lambda_distortion = 1e-4
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, results, target, **kwargs):
        d = {}
        for i in range(kwargs['stages']):
            # if kwargs['skip'] and i>0:
            #     break
            # if i==1:
            #     import ipdb; ipdb.set_trace()
            if kwargs.get('distill', False):
                labels = torch.argmax(results[f'semantic{i}'].detach(), dim=-1)
                mask = torch.logical_and(labels==1, labels==5)
                mask = torch.logical_and(labels==6, mask)
                mask = ~mask
                weight = torch.zeros_like(mask, dtype=torch.float32)
                prob_raw, _ = nn.functional.softmax(results[f'semantic{i}'].detach(), dim=-1).max(-1)
                weight[mask] = prob_raw[mask]
                d[f'rgb{i}'] = weight.unsqueeze(-1) * (results[f'rgb{i}']-target['rgb'])**2
            else: 
                d[f'rgb{i}'] = (results[f'rgb{i}']-target['rgb'])**2
            
            o = results[f'opacity{i}']+1e-10
            # encourage opacity to be either 0 or 1 to avoid floater
            # d[f'opacity{i}'] = self.lambda_opa*(-o*torch.log(o))
            # d[f'opacity{i}'] = self.lambda_opa*(1-o)**2
            
                
            if i==0 and i<kwargs['stages']-1:
                # for (i, n) in d.items():
                #     d[i] *= 0.1
                continue
            # if i==1 and i<kwargs['stages']-1:
            #     for (i, n) in d.items():
            #         d[i] *= 0.8
            if self.lambda_distortion > 0:
                d[f'distortion{i}'] = self.lambda_distortion*distortion_loss(results[f'ws{i}'], results[f'z_vals{i}'].detach())
            if kwargs.get('semantic', False):
                d[f'CELoss{i}'] = 4e-2*self.CrossEntropyLoss(results[f'semantic{i}'], target['label'])
                sky_mask = torch.where(target['label']==4, 1., 0.)
                d[f'sky_depth{i}'] = 1e-1*sky_mask*torch.exp(-results[f'depth{i}'])
            
            if kwargs.get('normal_p', False):
                d[f'Rp{i}'] = (results[f'Rp{i}']-torch.zeros_like(results[f'Rp{i}']).cuda()) * 2e-4
            
            
            
            
            flag = 0
            for (i, n) in d.items():
                if torch.any(torch.isnan(n)):
                    print(f'nan in d[{i}]')
                    print(f'max: {torch.max(n)}')
                    flag = 1
            
            # assert flag == 0, 'nan occurs!'
            
        return d
