from logging.config import valid_ident
from turtle import distance
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren
# from torchviz import make_dot
import trimesh
import numpy as np
import os

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
prev = {}
# Radius = 9e-3
N_samples = 100
center_density = 1000

def wrap_light(L, dir, wrap=20):
    '''
    Input:
    L: normalized light direction
    dir: direction related to the metaball center
    
    Output:
    diffuse_scale: a grey scale for diffuse color
    '''
    dot = torch.sum(L[None, :] * dir, dim=-1)
    diffuse_scale = (dot+wrap)/(1.+wrap)
    
    return diffuse_scale

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

def kernel_function(density_o, R, r):
    '''
    calculate densities for points inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    density_r: densities of samples (N_samples, ) 
    '''
    r = torch.clamp(r, max=R)
    # density_r = (R/0.002)**(1/3) * (-4/9*(r/R)**6+17/9*(r/R)**4-22/9*(r/R)**2+1)*density_o
    density_r = 315/(64*torch.pi*1.5**7)*(1.5**2-(r/R)**2)**3*density_o
    density_r = torch.clamp(density_r, min=0)
    return density_r

def dkernel_function(density_o, R, r):
    '''
    calculate derivatives for densities inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    ddensity_dr: derivatives of densities of samples (N_samples, ) 
    '''
    r = torch.clamp(r, max=R)
    # ddensity_dr = (R/0.002)**(1/3) * (-24/9*(r/R)**5+68/9*(r/R)**3-44/9*(r/R))*density_o
    ddensity_dr = -6*315/(64*torch.pi*1.5**7)*(1.5**2-(r/R)**2)**2*(r/R**2)*density_o
    ddensity_dr = torch.clamp(ddensity_dr, max=-1e-4)
    return ddensity_dr

def remapping_3d(up, xy_2d, R, height):
    '''
    Height denotes the height for every uv point
    Note: height should be the absolute height between sample and original point (related height+ground height).
    
    Input:
    up (3,)
    xy_2d (N, 2)
    R (3, 3)
    height (N,)
    
    Output:
    xyzs (N, 3)    
    '''
    u, v = xy_2d.unbind(-1)
    zeros = torch.zeros_like(u)
    xyzs= torch.stack([u, zeros, v], -1).reshape(-1, 3, 1)
    xyzs = torch.matmul(R, xyzs).reshape(-1, 3) + up[None, :]*height[:, None]

    return xyzs

def remapping_2d(up, xyz, R_inv):
    '''
    Remap 3D coordinates into 2D uv coordinate on a plane which includes original point
    
    Input:
    up (3,)
    xyz (N, 3)
    R_inv (3, 3)
    
    Output:
    uv (N, 2)
    '''
    normal_xyz = torch.sum(xyz*up[None, :], dim=-1, keepdim=True)*up[None, :]
    parallel_xyz = xyz - normal_xyz
    uv = torch.matmul(R_inv, parallel_xyz.reshape(-1, 3, 1)).reshape(-1, 3)
    u = uv[:, 0]
    v = uv[:, 2]
    
    return torch.stack([u,v], dim=-1)

def remapping_height(up, xyz, ground_height):
    '''
    Return the distance between samples and the ground plane
    
    Input:
    up (3,)
    xyz (N, 3)
    ground_height (1)
    
    Output:
    height (N,)
    '''
    height = torch.sum(xyz*up[None, :], dim=-1)
    height = height - ground_height
    
    return height

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
    if kwargs['view_from']=='sky':
        depth = torch.clamp(depth, max = torch.abs(kwargs['sky_height']-kwargs['ground_height']))
    
    centers = rays_o + rays_d*depth.unsqueeze(-1)
    centers_out = centers[~up_mask]
    centers = centers[up_mask]
    rgb = rgb[up_mask]
    # with torch.no_grad():
    #     density_o = model.proxy_density(centers) # density for the center of metaballs
    
    N_metaballs = centers.shape[0]
    density_o = center_density*torch.ones((N_metaballs)).cuda()
    
    return density_o, centers, centers_out, rgb, up_mask

def render(model, model_mb, rays_o, rays_d, **kwargs):
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

    results = render_func(model, model_mb, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, model_mb, rays_o, rays_d, hits_t, **kwargs):
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
    weight = torch.zeros(N_rays, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4
    
    # xyz_vis = []
    density_check_0 = []
    density_check_1 = []
    density_check_2 = []
    density_check_3 = []
    
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
        
        # height = torch.sum(xyzs[valid_mask]*kwargs['up_vector'][None, :], dim=-1, keepdim=True)
        _sigmas, _rgbs, _normals_pred, _sems = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        # _rgbs = torch.zeros_like(_rgbs).cuda()
        # _sigmas, _rgbs, _normals_pred, _sems = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        # _sigmas_mb, _rgbs_mb = model.forward_mb(xyzs[valid_mask], height)
        
        # radius = kwargs['interval'] / 4
        # sample_2d = remapping_2d(kwargs['up_vector'], xyzs[valid_mask], kwargs['R_inv'])
        # mask_in_box_1 = torch.logical_and(sample_2d[:, 0]<8, sample_2d[:, 0]>-8)
        # mask_in_box_2 = torch.logical_and(sample_2d[:, 1]<8, sample_2d[:, 1]>-8)
        # mask_in_box = torch.logical_and(mask_in_box_1, mask_in_box_2)
        # _sigmas_mb = torch.zeros((xyzs[valid_mask].shape[0])).cuda()
        # _rgbs_mb = torch.ones((xyzs[valid_mask].shape[0], 3)).cuda()
        
        # sample_2d_in_box = sample_2d[mask_in_box]
        # sample_in_box_u = torch.floor(sample_2d_in_box[:, 0]/radius)*radius
        # sample_in_box_v = torch.floor(sample_2d_in_box[:, 1]/radius)*radius
        # uv_0 = torch.stack([sample_in_box_u, sample_in_box_v], -1)
        # uv_1 = torch.stack([sample_in_box_u, sample_in_box_v+radius], -1)
        # uv_2 = torch.stack([sample_in_box_u+radius, sample_in_box_v], -1)
        # uv_3 = torch.stack([sample_in_box_u+radius, sample_in_box_v+radius], -1)
        # uv = torch.cat([uv_0, uv_1, uv_2, uv_3], dim=0)
        # nearest_height, nearest_density = model.forward_mb(uv)
        # nearest_height += kwargs['ground_height']+1e-3
        # nearest_height_0, nearest_height_1, nearest_height_2, nearest_height_3 = nearest_height.chunk(4)
        # nearest_density_0, nearest_density_1, nearest_density_2, nearest_density_3 = nearest_density.chunk(4)
        
        # nearest_xyz_0 = remapping_3d(kwargs['up_vector'], uv_0, kwargs['R'], nearest_height_0)
        
        # # points_ = nearest_xyz_0.cpu().numpy()
        # # color = np.ones((points_.shape[0], 4))
        # # point_cloud = trimesh.points.PointCloud(points_, color)
        # # point_cloud.export(os.path.join(f'logs/tnt/Playground_snow/test_samples.ply'))
        
        # nearest_xyz_1 = remapping_3d(kwargs['up_vector'], uv_1, kwargs['R'], nearest_height_1)
        # nearest_xyz_2 = remapping_3d(kwargs['up_vector'], uv_2, kwargs['R'], nearest_height_2)
        # nearest_xyz_3 = remapping_3d(kwargs['up_vector'], uv_3, kwargs['R'], nearest_height_3)
        
        # nearest_dist_0 = torch.linalg.norm(nearest_xyz_0-xyzs[valid_mask][mask_in_box], dim=-1)
        # nearest_dist_1 = torch.linalg.norm(nearest_xyz_1-xyzs[valid_mask][mask_in_box], dim=-1)
        # nearest_dist_2 = torch.linalg.norm(nearest_xyz_2-xyzs[valid_mask][mask_in_box], dim=-1)
        # nearest_dist_3 = torch.linalg.norm(nearest_xyz_3-xyzs[valid_mask][mask_in_box], dim=-1)
        
        # density_0 = kernel_function(center_density, radius, nearest_dist_0)
        # density_1 = kernel_function(center_density, radius, nearest_dist_1)
        # density_2 = kernel_function(center_density, radius, nearest_dist_2)
        # density_3 = kernel_function(center_density, radius, nearest_dist_3)
        # density_check_0.append(density_0)
        # density_check_1.append(density_1)
        # density_check_2.append(density_2)
        # density_check_3.append(density_3)
        # # density_0 = kernel_function(nearest_density_0, radius, nearest_dist_0)
        # # density_1 = kernel_function(nearest_density_1, radius, nearest_dist_1)
        # # density_2 = kernel_function(nearest_density_2, radius, nearest_dist_2)
        # # density_3 = kernel_function(nearest_density_3, radius, nearest_dist_3)
        
        # _sigmas_mb[mask_in_box] = (density_0+density_1+density_2+density_3)
        
        _sigmas_mb, _rgbs_mb = model_mb.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        # _rgbs_mb = torch.ones_like(_rgbs_mb).cuda() * _sigmas_mb[:, None] / (center_density*3)
        # _sigmas = _sigmas_mb
        # _rgbs = _rgbs_mb
        # _sigmas_mb, _rgbs_mb = model_mb(xyzs[valid_mask])
        # _sigmas = torch.zeros_like(_sigmas_mb).cuda()
        # _rgbs = torch.zeros_like(_rgbs_mb).cuda()
        # iso_mask = _sigmas_mb > center_density / 6
        # _sigmas[iso_mask] = _sigmas_mb[iso_mask]
        # _rgbs[iso_mask] = _rgbs_mb[iso_mask]
        weighted_sigmoid = lambda x, weight, bias : 1./(1+torch.exp(-weight*(x-bias)))
        thres_ratio = kwargs.get('thres', 1/8)
        thres = weighted_sigmoid(_sigmas_mb, 50, center_density * thres_ratio).detach()
        _sigmas = _sigmas * (1-thres) + _sigmas_mb * thres
        _rgbs = _rgbs * (1-thres[:, None]) + _rgbs_mb * thres[:, None]
        
        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
        # if kwargs.get('render_sem', False):
        #     sems[valid_mask] = _sems.float()
        # if kwargs.get('render_normal', False):
        #     normals_pred[valid_mask] = _normals_pred.float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_mb_test_fw(
            sigmas, rgbs, normals_pred, sems, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, sem, weight)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb # (h*w)
    results['normal_pred'] = normal_pred
    results['total_samples'] = total_samples # total samples for all rays
    results['weight'] = weight
    
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

    # xyz_vis = torch.cat(xyz_vis, dim=0).cpu().numpy()
    # color_ = np.ones((xyz_vis.shape[0], 4))
    # point_cloud = trimesh.points.PointCloud(xyz_vis, color_)
    # point_cloud.export(os.path.join(f'logs/tnt/Playground_snow/render_samples.ply'))
        
    return results


# @torch.cuda.amp.autocast()
def __render_rays_train(model, model_mb, rays_o, rays_d, hits_t, **kwargs):
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
        rays_a, xyzs, dirs, deltas, ts, total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    # results['total_samples'] = total_samples
    # import ipdb; ipdb.set_trace()
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor) and k not in ['up_vector', 'samples', 'embedding_a', 'ground_height', 'sky_height', 'R', 'R_inv']:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']
        kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)
    
    weighted_sigmoid = lambda x, weight, bias : 1./(1+torch.exp(-weight*(x-bias)))
    if kwargs.get('stylize', False):
        sigmas, results['rgbs_0'], normals_pred, semantics = model.forward_test(xyzs, dirs, **kwargs)
        sigmas_mb, results['rgbs_mb'] = model_mb.forward_test(xyzs, dirs, **kwargs)
        thres = weighted_sigmoid(sigmas_mb, 50, center_density / 8).detach()
        sigmas = sigmas * (1-thres) + sigmas_mb * thres
        rgbs = results['rgbs_0'] * (1-thres[:, None]) + results['rgbs_mb'] * thres[:, None]
        
        place_holder = torch.zeros_like(rgbs, requires_grad=False).cuda()
        up_sems = torch.zeros_like(sigmas, requires_grad=False).cuda()
        _, _, results['depth'], results['rgb'], normal_pred, _, _, results['semantic'], _ = \
            VolumeRenderer.apply(sigmas.contiguous(), rgbs.contiguous(), normals_pred.contiguous(), place_holder.contiguous(), up_sems.contiguous(), 
                                semantics.contiguous(), deltas, ts,
                                rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))        
        _, _, _, results['rgb_0'], _, _, _, _, _ = \
            VolumeRenderer.apply(sigmas.contiguous(), results['rgbs_0'].contiguous(), normals_pred.contiguous(), place_holder.contiguous(), up_sems.contiguous(), 
                                semantics.contiguous(), deltas, ts,
                                rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))        
    else:
        with torch.no_grad():
            sigmas, rgbs, normals_pred, semantics = model.forward_test(xyzs, dirs, **kwargs)
        with torch.no_grad():
            alphas, ws = vren.composite_alpha_fw(sigmas.contiguous(), deltas.contiguous(), rays_a.contiguous(), kwargs.get('T_threshold', 1e-4))

        cos_theta = torch.sum(kwargs['up_vector'][None, :]*normals_pred, dim=-1) # for every sample
        semantic_mask = torch.argmax(semantics, dim=-1) != 4
        
        results['alphas'] = (ws * weighted_sigmoid(cos_theta, 50, 0.85).detach() * semantic_mask.float()).detach() # work as g.t.
        # results['alphas'] = alphas.detach()
        results['alphas_mb'], results['rgbs'] = model_mb(xyzs, dirs) # 3D hash table
    # mask = results['alphas'] > 0.2
    # points_ = xyzs[mask].detach().cpu().numpy()
    # colors_ = np.ones((points_.shape[0], 4))
    # colors_ = colors_ * results['alphas'][mask].reshape(-1, 1).cpu().numpy()
    # colors_[..., 3] = 1.
    # point_cloud = trimesh.points.PointCloud(points_, colors_)
    # point_cloud.export(os.path.join(f'logs/tnt/Playground_snow/test_alpha.ply'))
    # import ipdb; ipdb.set_trace()
    # sample_height = torch.sum(kwargs['samples'] * kwargs['up_vector'][None, :], dim=-1, keepdim=True)
    # results['density_r_pred'], _ = model.forward_mb(kwargs['samples'], sample_height)
    # results['depth'] = depth
    # depth_bg = depth.max()
    # results['depth'] += depth_bg * (1-opacity) # for regularization on depth, might not be useful
    for (i, n) in results.items():
        if torch.any(torch.isnan(n)):
            print(f'nan in results[{i}]')
        if torch.any(torch.isinf(n)):
            print(f'inf in results[{i}]')
                    
    return results