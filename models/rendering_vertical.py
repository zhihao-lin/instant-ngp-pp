from logging.config import valid_ident
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren
# from torchviz import make_dot

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
height = -0.1
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
    classes = kwargs.get('classes', 7)
    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    normal_pred = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)
    up_sem = torch.zeros(N_rays, device=device)
    sem = torch.zeros(N_rays, classes, device=device)

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
        normals_pred = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        up_sems = torch.zeros(len(xyzs), device=device)
        sems = torch.zeros(len(xyzs), classes, device=device)
        if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
            kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
       
        # _sigmas, _rgbs, _normals_raw, _normals_pred, _sems, _ = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        _sigmas, _rgbs, _normals_pred, _sems = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        
        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
        # if kwargs.get('render_up_sem', False):
        #     up_sems[valid_mask] = _up_sems.float()
        if kwargs.get('render_sem', False):
            sems[valid_mask] = _sems.float()
        if kwargs.get('render_normal', False):
            # normals_raw[valid_mask] = _normals_raw.float()
            normals_pred[valid_mask] = _normals_pred.float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        up_sems = rearrange(up_sems, '(n1 n2) -> n1 n2', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_test_fw(
            sigmas, rgbs, normals_pred, normals_raw, up_sems, sems, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, normal_raw, up_sem, sem)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb # (h*w)
    results['normal_pred'] = normal_pred
    results['normal_raw'] = normal_raw
    results['total_samples'] = total_samples # total samples for all rays
        
    up = F.normalize(kwargs['up'], dim=0)
    proxy = rays_o + rays_d*depth[:, None]
    principle_xyz = height*up
    # principle_xyz = kwargs['ground_height']*up
    related = proxy - principle_xyz
    related_cos = torch.sum(related * up, dim=-1)
    mask = related_cos<0
    # import ipdb; ipdb.set_trace()
    t_intersect = torch.abs(torch.sum((principle_xyz[None, :]-rays_o)*up[None, :], dim=-1)/torch.sum(rays_d*up[None, :], dim=-1))
    
    # surface_xyz = rays_o + rays_d*t_intersect[:, None]
    # # results['surface'] = proxy
    # results['surface'] = surface_xyz[mask]

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=device)
    if kwargs.get('use_skybox', False):
        # print('rendering skybox')
        rgb_bg = model.forward_skybox(rays_d)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')
    if kwargs.get('render_up_sem', False):
        results['rgb+up'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')
    
    if torch.any(mask):
        results['rgb'][mask] = 0.5
        results['depth'][mask] = t_intersect[mask].float()
        results['normal_pred'][mask] = up.float()
        
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
        rays_a, xyzs, dirs, results['deltas'], results['ts'], total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples
    # import ipdb; ipdb.set_trace()
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor) and k not in ['up', 'ground_height']:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
        
    with torch.no_grad():
        sigmas, rgbs, normals_pred, sems = model.forward_test(xyzs, dirs, **kwargs)
    # make_dot(sigmas, params=dict(model.named_parameters()))
    results['sigma'] = sigmas
    results['xyzs'] = xyzs
    results['rays_a'] = rays_a
    
    # import ipdb; ipdb.set_trace()
    place_holder = torch.zeros_like(rgbs, requires_grad=False).cuda()
    up_sems = torch.zeros_like(sigmas, requires_grad=False).cuda()
    with torch.no_grad():
        _, opacity, depth, rgb, normal_pred, _, _, sem, _ = \
            VolumeRenderer.apply(sigmas, rgbs.contiguous(), normals_pred.contiguous(), place_holder.contiguous(), up_sems.contiguous(), 
                                    sems.contiguous(), results['deltas'], results['ts'],
                                    rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))
    
    results['semantic'] = torch.argmax(sem, dim=-1, keepdim=True)
    mask = results['semantic']==0
    mask = mask.squeeze(-1)
    normal_pred = F.normalize(normal_pred, dim=-1)
    results['normal'] = normal_pred[mask].detach()
    
    depth += depth.max()*(1-opacity)
    results['depth'] = depth[mask]
    results['xyz'] = (rays_o + rays_d*depth[:, None])[mask]
    results['rgb'] = rgb[mask]
    
    up = kwargs['up']
    results['vertical_dist'] = torch.abs(torch.sum((results['xyz']-kwargs['ground_height']*up[None, :])*up[None, :], dim=-1))
    
    for (i, n) in results.items():
        if torch.any(torch.isnan(n)):
            print(f'nan in results[{i}]')
        if torch.any(torch.isinf(n)):
            print(f'inf in results[{i}]')

    return results