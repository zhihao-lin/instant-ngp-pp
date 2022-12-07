from logging.config import valid_ident
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
prev = {}

def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
    
    if kwargs.get('depth_smooth', False):
        rays_d_xdx = kwargs['rays_d_xdx'].contiguous()
        rays_d_ydy = kwargs['rays_d_ydy'].contiguous()
        _, hits_t_xdx, _ = \
            RayAABBIntersector.apply(rays_o, rays_d_xdx, model.center, model.half_size, 1)
        hits_t_xdx[(hits_t_xdx[:, 0, 0]>=0)&(hits_t_xdx[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
        _, hits_t_ydy, _ = \
            RayAABBIntersector.apply(rays_o, rays_d_ydy, model.center, model.half_size, 1)
        hits_t_ydy[(hits_t_ydy[:, 0, 0]>=0)&(hits_t_ydy[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
        kwargs['rays_d_xdx'] = rays_d_xdx
        kwargs['rays_d_ydy'] = rays_d_ydy
        kwargs['hits_t_xdx'] = hits_t_xdx
        kwargs['hits_t_ydy'] = hits_t_ydy

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']
        # kwargs['embedding_a'] = torch.repeat_interleave(kwargs['embedding_a'], rays_o.shape[0], 0)

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    normal = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        normals = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
            kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
       
        # with torch.enable_grad():
        # import ipdb; ipdb.set_trace()
        _sigmas, _rgbs, _normals = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)
            # _sigmas, _rgbs, _, _normals = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        with torch.enable_grad():
            xyzs_ = xyzs[valid_mask]
            xyzs_ = xyzs_.requires_grad_(True)
            _sigmas_raw = model.density(xyzs_)
            grads_raw = torch.autograd.grad(outputs=_sigmas_raw,
                                            inputs=xyzs_,
                                            grad_outputs=torch.ones_like(_sigmas_raw, requires_grad=False))[0]
            grads_raw = grads_raw.detach()
        _normals_raw = -F.normalize(grads_raw, p=2, eps=1e-9)
        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
            normals[valid_mask] = _normals.detach().float()
            normals_raw[valid_mask] = _normals_raw.detach().float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals = rearrange(normals, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        
        vren.composite_test_fw(
            sigmas, rgbs, normals, normals_raw, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb, normal, normal_raw)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb # (h*w)
    results['total_samples'] = total_samples # total samples for all rays
    results['normal_raw'] = F.normalize(normal_raw, p=2, dim=-1, eps=1e-6)
    results['normal'] = F.normalize(normal, p=2, dim=-1, eps=1e-6)

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=device)
    elif kwargs.get('use_skybox', False):
        rgb_bg = model.forward_skybox(rays_d)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    return results


# @torch.cuda.amp.autocast()
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        rays_a, xyzs, dirs, deltas, ts, total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples
    
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    
    if kwargs.get('distill_normal', False):
        normal_model = kwargs['normal_model']
        
        with torch.enable_grad():
            xyzs.requires_grad_(True)
            sigma_raw = model.density(xyzs)
            results['sigma'] = sigma_raw
            results['xyzs'] = xyzs
              
            rgbs = torch.zeros_like(xyzs).cuda()
            normals_pred = torch.zeros_like(xyzs).cuda()
            # normals_pred = normal_model(xyzs)
            
            normals_raw = -torch.autograd.grad(
                            outputs=sigma_raw,
                            inputs=xyzs,
                            grad_outputs=torch.ones_like(sigma_raw, requires_grad=False),
                            retain_graph=True
                            )[0]
            
            normals_raw = F.normalize(normals_raw, p=2, dim=-1, eps=1e-6).detach()
        
        results['opacity'], results['depth'], _, _, _, results['normal_raw'] = \
                VolumeRenderer.apply(sigma_raw, rgbs.contiguous(), normals_pred.contiguous(), normals_raw.contiguous(), deltas, ts,
                                    rays_a, kwargs.get('T_threshold', 1e-4))
        
        valid_mask = results['opacity']>0.99
        surface_x = results['depth'][valid_mask].unsqueeze(-1)*rays_d[valid_mask] + rays_o[valid_mask]
        surface_x = surface_x.detach()
        results['normal_pred'] = normal_model(surface_x)
        epsilon = torch.normal(mean=torch.zeros_like(surface_x), std=0.05*torch.ones_like(surface_x)).cuda()
        results['normal_pred_e'] = normal_model(epsilon+surface_x)
        results['normal_pred'] = F.normalize(results['normal_pred'], p=2, dim=-1, eps=1e-6)
        results['normal_raw'] = F.normalize(results['normal_raw'][valid_mask], p=2, dim=-1, eps=1e-6)
    else:
        # print(xyzs)
        # import ipdb; ipdb.set_trace()
        sigmas, rgbs, normals, normals_pred = model(xyzs, dirs, **kwargs)
        normals = normals.detach()
        # sigmas = torch.nan_to_num(sigmas)
        results['sigma'] = sigmas
        results['xyzs'] = xyzs
        results['rays_a'] = rays_a
        # results['normals'] = normals
        # results['normals_pred'] = normals_pred
        # if torch.any(torch.isnan(normals)):
        #     print('normals contains nan!')
        #     import ipdb; ipdb.set_trace()
        # if torch.any(torch.isnan(normals_pred)):
        #     print('normals_pred contains nan!')
        # mask = torch.logical_or(torch.isnan(normals), torch.isinf(normals))
        # normals = torch.nan_to_num(normals, 1e-6)
        # # normals_pred[~mask] = 0.
        # normals_pred = torch.where(mask, 1e-6, normals_pred)
                
        place_holder = torch.zeros_like(rgbs, requires_grad=False).cuda()
        results['opacity'], results['depth'], _, results['rgb'], results['normal_pred'], _ = \
            VolumeRenderer.apply(sigmas, rgbs.contiguous(), normals_pred.contiguous(), place_holder.contiguous(), deltas, ts,
                                rays_a, kwargs.get('T_threshold', 1e-4))

        # normals_diff = torch.zeros_like(normals).cuda()
        normals_diff = (normals-normals_pred)**2
        results['normals_diff'] = normals_diff
        # normals_diff = 1-torch.sum(normals * normals_pred, dim=-1)
        dirs = F.normalize(dirs, p=2, dim=-1, eps=1e-6)
        normals_ori = torch.clamp(torch.sum(normals_pred*dirs, dim=-1), min=0.)**2 # don't keep dim!
        results['normals_ori'] = normals_ori
        # normals_ori = torch.zeros(normals_pred.shape[0]).cuda()
        # mask = torch.sum(normals_pred*dirs, dim=-1)>0
        # normals_ori[mask] = (torch.sum(normals_pred*dirs, dim=-1)**2)[mask]
        
        results['Ro'], results['Rp'] = \
            RefLoss.apply(sigmas, normals_diff.contiguous(), normals_ori.contiguous(), deltas, ts,
                                rays_a, kwargs.get('T_threshold', 1e-4))
        # import ipdb; ipdb.set_trace()
        # alpha = 1-torch.exp(-deltas*sigmas)
        # results['Rp'] = torch.mean((alpha.view(-1,1) * normals_diff).sum(-1))
        # results['Ro'] = torch.mean((alpha * normals_ori))
        # results['Ro'] = torch.mean(results['Ro'])
        # results['Rp'] = torch.mean(results['Rp'].sum(-1))
        
        if kwargs.get('depth_smooth', False):
            with torch.no_grad():
                rays_a1, xyzs1, dirs1, deltas1, ts1, total_samples1 = \
                    RayMarcher.apply(
                        rays_o, kwargs['rays_d_xdx'], kwargs['hits_t_xdx'][:, 0], model.density_bitfield,
                        model.cascades, model.scale,
                        exp_step_factor, model.grid_size, MAX_SAMPLES)
                rays_a2, xyzs2, dirs2, deltas2, ts2, total_samples2 = \
                    RayMarcher.apply(
                        rays_o, kwargs['rays_d_ydy'], kwargs['hits_t_ydy'][:, 0], model.density_bitfield,
                        model.cascades, model.scale,
                        exp_step_factor, model.grid_size, MAX_SAMPLES)
                sigmas1 = model.density(xyzs1)
                sigmas2 = model.density(xyzs2)
                place_holder1 = torch.zeros_like(xyzs1, requires_grad=False).cuda()
                place_holder2 = torch.zeros_like(xyzs2, requires_grad=False).cuda()
                _, results['depth1'], _, _, _, _ = \
                VolumeRenderer.apply(sigmas1, place_holder1.contiguous(), place_holder1.contiguous(), place_holder1.contiguous(), deltas1, ts1,
                                    rays_a1, kwargs.get('T_threshold', 1e-4))
                _, results['depth2'], _, _, _, _ = \
                VolumeRenderer.apply(sigmas2, place_holder2.contiguous(), place_holder2.contiguous(), place_holder2.contiguous(), deltas2, ts2,
                                    rays_a2, kwargs.get('T_threshold', 1e-4))
            
        if kwargs.get('use_skybox', False):
            rgb_bg = model.forward_skybox(rays_d)
        elif exp_step_factor==0: # synthetic
            if kwargs.get('random_bg', False):
                rgb_bg = torch.rand(3, device=rays_o.device)
            rgb_bg = torch.ones(3, device=rays_o.device)
        else: # real
            if kwargs.get('random_bg', False):
                rgb_bg = torch.rand(3, device=rays_o.device)
            else:
                rgb_bg = torch.zeros(3, device=rays_o.device)
        
        results['rgb'] = results['rgb'] + \
                    rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')
        
        for (i, n) in results.items():
            if torch.any(torch.isnan(n)):
                print(f'nan in results[{i}]')
        #         import ipdb; ipdb.set_trace()
        # prev['sigma'] = sigmas
        # prev['rgb'] = rgbs
        # prev['normals'] = normals
        # prev['normals_pred'] = normals_pred
        # prev['normals_diff'] = normals_diff
        # prev['normals_ori'] = normals_ori

    return results