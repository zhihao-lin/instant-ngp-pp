import os
import shutil
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path
import skimage.io
import tqdm
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    # parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    patch_size = 8
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--dataset_path", default="/media/yuan/3e33a457-e1df-4867-b421-21cadc1473bc/3d_mass_estimation/dataset/HM3D-ABO/author_shared/scenes", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()

    for scene_path in tqdm.tqdm(Path(args.dataset_path).glob("*"), total=len(list(Path(args.dataset_path).glob("*")))):
        os.makedirs(str(scene_path / "dino_features"), exist_ok=True)
        shutil.rmtree(str(scene_path / "rgb_bg"))
        for img_path in sorted(list((scene_path / "rgb").glob("*.jpg"))):
            with torch.no_grad():
                # open image
                if not os.path.exists(str(img_path).replace('rgb', 'dino_features').replace('jpg', 'npy')):
                    with open(str(img_path), 'rb') as f:
                        img = Image.open(f)
                        img = img.convert('RGB')

                    transform = pth_transforms.Compose([
                        pth_transforms.CenterCrop(min(img.size[:2])),
                        pth_transforms.Resize(args.image_size),
                        pth_transforms.ToTensor(),
                        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                    img = transform(img)

                    # make the image divisible by the patch size
                    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
                    img = img[:, :w, :h].unsqueeze(0)

                    w_featmap = img.shape[-2] // patch_size
                    h_featmap = img.shape[-1] // patch_size

                    # attentions = model.get_last_selfattention(img.to(device))
                    dino_features = model.get_intermediate_layers(img.to(device))[0][0, 1:]

                    # nh = attentions.shape[1] # number of head

                    # we keep only the output patch attention
                    # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

                    # if args.threshold is not None:
                    #     # we keep only a certain percentage of the mass
                    #     val, idx = torch.sort(attentions)
                    #     val /= torch.sum(val, dim=1, keepdim=True)
                    #     cumval = torch.cumsum(val, dim=1)
                    #     th_attn = cumval > (1 - args.threshold)
                    #     idx2 = torch.argsort(idx)
                    #     for head in range(nh):
                    #         th_attn[head] = th_attn[head][idx2[head]]
                    #     th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                    #     # interpolate
                    #     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
                    #
                    # attentions = attentions.reshape(nh, w_featmap, h_featmap)
                    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

                    dino_features = dino_features.permute(1, 0).view(-1, w_featmap, h_featmap).cpu().numpy()
                    np.save(str(img_path).replace('rgb', 'dino_features').replace('jpg', 'npy'), dino_features)
                    # dino_features = nn.functional.interpolate(dino_features.unsqueeze(0), scale_factor=(patch_size, patch_size), mode="bilinear")[0].cpu().numpy()
                    #
                    # # save attentions heatmaps
                    # os.makedirs(args.output_dir, exist_ok=True)
                    # torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
                    # for j in range(nh):
                    #     fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
                    #     plt.imsave(fname=fname, arr=attentions[j], format='png')
                    #     print(f"{fname} saved.")
                    #
                    # if args.threshold is not None:
                    #     image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
                    #     for j in range(nh):
                    #         display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
