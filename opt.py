import argparse
import configargparse

def get_opts():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # common args for all datasets
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nerf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'tnt', 'kitti', 'mega'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    parser.add_argument('--anti_aliasing_factor', type=float, default=1.0,
                        help='Render larger images then downsample')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_skybox', action='store_true', default=False,
                        help='whether to use skybox')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    parser.add_argument('--embed_a', action='store_true', default=False,
                        help='whether to use appearance embeddings')
    parser.add_argument('--embed_a_len', type=int, default=4,
                        help='the length of the appearance embeddings')
    parser.add_argument('--embed_msk', action='store_true', default=False,
                        help='whether to use sigma embeddings')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='total number of semantic classes')

    # for kitti 360 dataset
    parser.add_argument('--kitti_seq', type=int, default=0, 
                        help='scene sequence index')
    parser.add_argument('--kitti_start', type=int, default=1538,
                        help='starting frame index')
    parser.add_argument('--kitti_end', type=int, default=1601,
                        help='ending frame index')
    parser.add_argument('--kitti_test_id', type=int, nargs='+', default=[],
                        help='frames for testing')
    parser.add_argument('--nvs', action='store_true', default=False,
                        help='Evaluate NVS and exclude testing frames in training')

    # for mega-nerf dataset
    parser.add_argument('--mega_frame_start', type=int, default=753)
    parser.add_argument('--mega_frame_end',   type=int, default=846)

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--normal_epochs', type=int, default=20,
                        help='number of training epochs for normal distillation')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--depth_mono', action='store_true', default=False,
                        help='use 2D predicted depth')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics (experimental)')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--render_traj', action='store_true', default=False,
                        help='render video on a trajectory')
    parser.add_argument('--render_train', action='store_true', default=False,
                        help='interpolate among training views to get camera trajectory')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_load', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--ckpt_save', type=str, default='checkpoint.ckpt',
                        help='pretrained checkpoint to save (including optimizers, etc)')
    
    # render
    parser.add_argument('--render_rgb', action='store_true', default=False,
                        help='render rgb series')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='render depth series')
    parser.add_argument('--render_normal', action='store_true', default=False,
                        help='render normal series')
    parser.add_argument('--render_semantic', action='store_true', default=False,
                        help='render semantic segmentation series')
    parser.add_argument('--normal_composite', action='store_true', default=False,
                        help='render normal+rgb composition series')
    parser.add_argument('--render_points', action='store_true', default=False,
                        help='render depth points')
    parser.add_argument('--chunk_size', type=int, default=131072, 
                        help='Divide image into chunks for rendering')
    return parser.parse_args()
