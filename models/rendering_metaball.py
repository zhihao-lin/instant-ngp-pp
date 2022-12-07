from logging.config import valid_ident
from turtle import distance
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren
# from torchviz import make_dot

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
prev = {}
Radius = 9e-3
N_samples = 100
center_density = 500

def metaball_samples(center, R=1e-4, N_samples=100):
    '''
    sample points inside metaballs
    
    Inputs:
    center: centers of metaballs (N_balls, 3)
    R: radius of metaballs (float)
    N_samples: total number of samples inside metaballs (int)
    
    Output:
    samples: samples inside metaballs (N_balls, N_samples, 3)
    r: radius of each sample (N_balls, N_samples)
    '''
    N_balls = center.shape[0]
    theta = 2*torch.rand((N_balls, N_samples))*torch.pi
    phi = 2*torch.rand((N_balls, N_samples))*torch.pi
    r = torch.rand((N_balls, N_samples)) * R
    x = torch.sin(phi)*torch.cos(theta) * r
    y = torch.sin(phi)*torch.sin(theta) * r
    z = torch.cos(phi) * r
    xyz = torch.stack([x, y, z], dim=-1).cuda()
    samples = xyz + center.unsqueeze(1).repeat(1, N_samples, 1)
    return samples, torch.ones_like(r)*R, r

def metaball_halfsamples(center, R=1e-4, N_samples=100):
    '''
    sample points inside metaballs
    
    Inputs:
    center: centers of metaballs (N_balls, 3)
    R: radius of metaballs (float)
    N_samples: total number of samples inside metaballs (int)
    
    Output:
    samples: samples inside metaballs (N_balls, N_samples, 3)
    '''
    N_balls = center.shape[0]
    theta = 2*torch.rand((N_balls, N_samples))*torch.pi
    phi = 2*torch.rand((N_balls, N_samples))*torch.pi
    r = 0.5 * R
    x = torch.sin(phi)*torch.cos(theta) * r
    y = torch.sin(phi)*torch.sin(theta) * r
    z = torch.cos(phi) * r
    xyz = torch.stack([x, y, z], dim=-1).cuda()
    samples = xyz + center.unsqueeze(1).repeat(1, N_samples, 1)
    return samples

def metaball_sigmas(density_o, R, r):
    '''
    calculate densities for points inside metaballs
    
    Inputs:
    density_o: densities in the center (N_balls,)
    R: radius of metaballs (N_balls, N_samples)
    r: radius of samples inside metaballs (N_balls, N_samples)
    
    Output:
    density_r: densities of samples (N_balls, N_samples,) 
    '''
    N_balls, N_samples = r.shape
    density_r = (-4/9*(r/R)**6+17/9*(r/R)**4-22/9*(r/R)**2+1)*density_o.unsqueeze(-1).repeat(1,N_samples)
    return density_r

def metaball_seeds(model, rays_o, rays_d, **kwargs):
    # import ipdb; ipdb.set_trace()
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
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

    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor) and k not in ['up_vector', 'samples', 'embedding_a', 'ground_height', 'sky_height']:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
            
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']
        kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)
    
    with torch.no_grad():
        sigmas, rgbs, normals_pred, sems = model.forward_proxy(xyzs, dirs, **kwargs)

    place_holder = torch.zeros_like(rgbs, requires_grad=False).cuda()
    up_sems = torch.zeros_like(sigmas, requires_grad=False).cuda()
    with torch.no_grad():
        _, opacity, depth, rgb, normal_pred, _, _, semantic, _ = \
            VolumeRenderer.apply(sigmas, rgbs.contiguous(), normals_pred.contiguous(), place_holder.contiguous(), up_sems.contiguous(), 
                                    sems.contiguous(), results['deltas'], results['ts'],
                                    rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))    
    semantic_label = torch.argmax(semantic, dim=-1, keepdim=True)
    up_vector = torch.zeros_like(normal_pred)
    up_vector[..., :] = torch.Tensor(kwargs['up_vector']).cuda()
    up_mask = torch.matmul(up_vector[:, None, :], normal_pred[:, :, None]).squeeze(-1)\
            /(torch.linalg.norm(up_vector, dim=-1, keepdim=True)*torch.linalg.norm(normal_pred, dim=-1, keepdim=True)+1e-6)
    up_mask = torch.logical_and(up_mask>0.8, semantic_label!=4).squeeze(-1) # upward vectors while not sky
    
    depth_bg = depth.max()
    depth += depth_bg * (1-opacity)
    
    centers = rays_o + rays_d*depth.unsqueeze(-1)
    centers = centers[up_mask]
    rgb = rgb[up_mask]
    # with torch.no_grad():
    #     density_o = model.proxy_density(centers) # density for the center of metaballs
    
    N_metaballs = centers.shape[0]
    density_o = center_density*torch.ones((N_metaballs)).cuda()
    
    half_samples = metaball_halfsamples(centers, R=Radius, N_samples=N_samples).reshape(-1, 3) # sample points on half radius
    half_samples_height = torch.sum(half_samples*kwargs['up_vector'][None, :], dim=-1, keepdim=True)
    with torch.no_grad():
        density_half_o, _ = model.forward_mb(half_samples, half_samples_height)
    density_half_o = density_half_o.reshape((N_metaballs, N_samples))
    # to_add_sigma = density_half_o >= 0.5*density_o.unsqueeze(-1)
    to_add_sigma = density_half_o >= 0.5*center_density
    to_add_sigma_mask = (to_add_sigma.sum(-1).float()/N_samples) <= 0.8 # get rid of the metaballs that already change densities in surrounding spaces
    
    centers_out = centers[~to_add_sigma_mask]
    centers = centers[to_add_sigma_mask]
    centers_rgb = rgb[to_add_sigma_mask]
    # centers_out = torch.cat([centers_out0, centers_out1], dim=0)
    density_o = density_o[to_add_sigma_mask]
    samples, R, samples_r = metaball_samples(centers, R=Radius, N_samples=N_samples)
    samples_out, _, _ = metaball_samples(centers_out, R=Radius, N_samples=N_samples)
    density_r_cal = metaball_sigmas(density_o, R.cuda(), samples_r.cuda())
    
    samples = samples.reshape(-1, 3)
    samples_height = torch.sum(samples*kwargs['up_vector'][None, :], dim=-1, keepdim=True)
    with torch.no_grad():
        density_r_, _ = model.forward_mb(samples, samples_height)
    density_r_target = density_r_cal.reshape(-1) + density_r_ # calculate the sum of current metaball density and other metaball density
    
    return density_r_target, samples, depth.detach(), centers, samples_out.reshape(-1, 3), centers_rgb

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
    sem = torch.zeros(N_rays, classes, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    # while samples < kwargs.get('max_samples', MAX_SAMPLES):
    #     N_alive = len(alive_indices)
    #     if N_alive==0: break

    #     # the number of samples to add on each ray
    #     N_samples = max(min(N_rays//N_alive, 64), min_samples)
    #     samples += N_samples

    #     xyzs, dirs, deltas, ts, N_eff_samples = \
    #         vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
    #                               model.density_bitfield, model.cascades,
    #                               model.scale, exp_step_factor,
    #                               model.grid_size, MAX_SAMPLES, N_samples)
    #     total_samples += N_eff_samples.sum()
        
    #     xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
    #     dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        
    #     valid_mask = ~torch.all(dirs==0, dim=1)
    #     if valid_mask.sum()==0: break

    #     sigmas = torch.zeros(len(xyzs), device=device)
    #     rgbs = torch.zeros(len(xyzs), 3, device=device)
    #     normals_pred = torch.zeros(len(xyzs), 3, device=device)
    #     sems = torch.zeros(len(xyzs), classes, device=device)
    #     if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
    #         kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
       
    #     _sigmas, _rgbs, _normals_pred, _sems = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        
    #     if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
    #         sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
    #     if kwargs.get('render_sem', False):
    #         sems[valid_mask] = _sems.float()
    #     if kwargs.get('render_normal', False):
    #         normals_pred[valid_mask] = _normals_pred.float()
            
    #     sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
    #     rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
    #     normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
    #     sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
    #     vren.composite_mb_test_fw(
    #         sigmas, rgbs, normals_pred, sems, deltas, ts,
    #         hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4), classes,
    #         N_eff_samples, opacity, depth, rgb, normal_pred, sem)
    #     alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    # opacity_tmp = opacity
    # depth_tmp = depth
    # depth_tmp += (1-opacity_tmp)*depth.max()
    
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
        sems = torch.zeros(len(xyzs), classes, device=device)
        if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
            kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
       
        height = torch.sum(xyzs[valid_mask]*kwargs['up_vector'][None, :], dim=-1, keepdim=True)
        _sigmas, _rgbs, _normals_pred, _sems = model(xyzs[valid_mask], dirs[valid_mask], height, **kwargs)
        
        _sigmas_mb, _rgbs_mb = model.forward_mb(xyzs[valid_mask], height)
        iso_mask = _sigmas_mb > center_density/2
        _sigmas[iso_mask] = _sigmas_mb[iso_mask]
        _rgbs[iso_mask] = _rgbs_mb[iso_mask]
        
        # centers_o = torch.Tensor(kwargs['centers']).cuda()
        # sigmas_list = []
        # rgbs_list = []
        # chunk_size = 2048
        # masked_xyzs = xyzs[valid_mask]
        # for i in range(0, masked_xyzs.shape[0], chunk_size):
        #     _xyzs = masked_xyzs[i:i+chunk_size]
        #     __sigmas = _sigmas[i:i+chunk_size]
        #     __rgbs = _rgbs[i:i+chunk_size]
        #     dist_o = torch.sqrt(torch.sum((_xyzs[:, None, :] - centers_o[None, :, :])**2, dim=-1))
        #     dist_o_least, least_idx = torch.topk(dist_o, 5, dim=-1, largest=False)
        #     mask_o1 = dist_o_least<=Radius*0.79221
        #     in_ball_mask = torch.sum(mask_o1, dim=-1)>1
        #     # dist_o_ = dist_o[least_idx] # selected metaball dists
        #     mb_sigmas = torch.clamp(torch.sum((-4/9*(dist_o_least/Radius)**6+17/9*(dist_o_least/Radius)**4-22/9*(dist_o_least/Radius)**2+1)*center_density, dim=-1), min=0)
        #     iso_mask = mb_sigmas>0.5*center_density
        #     total_mask = torch.logical_and(in_ball_mask, iso_mask)
        #     __sigmas[total_mask] = mb_sigmas[total_mask]
        #     __rgbs[total_mask] = torch.ones_like(__rgbs).cuda()[total_mask]
        #     sigmas_list.append(__sigmas)
        #     rgbs_list.append(__rgbs)
        # _sigmas = torch.cat(sigmas_list, dim=0).cuda()
        # _rgbs = torch.cat(rgbs_list, dim=0).cuda()
        
        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
        if kwargs.get('render_sem', False):
            sems[valid_mask] = _sems.float()
        if kwargs.get('render_normal', False):
            normals_pred[valid_mask] = _normals_pred.float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_mb_test_fw(
            sigmas, rgbs, normals_pred, sems, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, sem)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb # (h*w)
    results['normal_pred'] = normal_pred
    results['total_samples'] = total_samples # total samples for all rays
    
    if kwargs.get('render_sem', False):
        mask = opacity<1e-2
        results['semantic'] = torch.argmax(sem, dim=-1, keepdim=True)
        results['semantic'][mask] = 4

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
    with torch.no_grad():
        rays_a, xyzs, dirs, results['deltas'], results['ts'], total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples
    
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor) and k not in ['up_vector', 'samples', 'embedding_a', 'ground_height', 'sky_height']:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']
        kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)
    
    # with torch.no_grad():
    # sigmas, rgbs, normals_pred, sems = model(xyzs, dirs, sample_height, **kwargs) # for regularization on depth, might not be useful

    # place_holder = torch.zeros_like(rgbs, requires_grad=False).cuda()
    # up_sems = torch.zeros_like(sigmas, requires_grad=False).cuda()
    # with torch.no_grad():
    #     _, opacity, depth, rgb, normal_pred, _, _, semantic, _ = \
    #         VolumeRenderer.apply(sigmas, rgbs.contiguous(), normals_pred.contiguous(), place_holder.contiguous(), up_sems.contiguous(), 
    #                                 sems.contiguous(), results['deltas'], results['ts'],
    #                                 rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))
    
    sample_height = torch.sum(kwargs['samples'] * kwargs['up_vector'][None, :], dim=-1, keepdim=True)
    results['density_r_pred'], _ = model.forward_mb(kwargs['samples'], sample_height)
    # results['depth'] = depth
    # depth_bg = depth.max()
    # results['depth'] += depth_bg * (1-opacity) # for regularization on depth, might not be useful
    for (i, n) in results.items():
        if torch.any(torch.isnan(n)):
            print(f'nan in results[{i}]')
        if torch.any(torch.isinf(n)):
            print(f'inf in results[{i}]')
                    
    return results