import torch
from torch import nn
import vren

class MetaballLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_opa = 1e-3
        self.lambda_distortion = 1e-4
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, results, target, **kwargs):
        d = {}
        # import ipdb; ipdb.set_trace()
        d['metaball_sigma'] = (kwargs['density_r_target']-results['density_r_pred'])**2
        # d['depth_reg'] = 1e-2*(kwargs['depth_proxy']-results['depth'])**2
        # d['sparsity_uniform'] = .001*(1.0-torch.exp(-0.001*kwargs['uniform_density']).mean())
            
        flag = 0
        for (i, n) in d.items():
            if torch.any(torch.isnan(n)):
                print(f'nan in d[{i}]')
                print(f'max: {torch.max(n)}')
                flag = 1
        
        # assert flag == 0, 'nan occurs!'
            
        return d
