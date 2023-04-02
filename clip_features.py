import torch
import clip
from PIL import Image
import numpy as np
import os
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_path = "/media/yuan/3e33a457-e1df-4867-b421-21cadc1473bc/3d_mass_estimation/dataset/HM3D-ABO/author_shared/scenes"
output_path = "/media/yuan/T7_red/VisualScale/dataset/HM3D_ABO/scenes"
for scene_path in tqdm.tqdm(Path(dataset_path).glob("*"), total=len(list(Path(dataset_path).glob("*")))):
    os.makedirs(output_path + f"/{scene_path.name}/clip_features", exist_ok=True)
    for img_path in sorted(list((scene_path / "rgb").glob("*.jpg"))):
        with torch.no_grad():
            # open image
            with open(str(img_path), 'rb') as f:
                img = Image.open(f)
                orig_image = img.convert('RGB')

            transform = pth_transforms.Compose([
                pth_transforms.CenterCrop(min(orig_image.size[:2])),
                pth_transforms.ToTensor(),
            ])
            orig_image = transform(orig_image)

            orig_image = torch.from_numpy(np.array(orig_image))[None,].cuda()
            with torch.no_grad():
                all_patch_features = []
                for ratio in np.linspace(start=0.05, stop=0.5, num=5):
                    patch_size = int(min(orig_image.shape[-2:]) * ratio)
                    if not os.path.exists(output_path + f"/{scene_path.name}/clip_features/{img_path.name[:-4]}_patch_size_{patch_size:03d}.npy"):
                        size = patch_size
                        stride = patch_size // 2 # LERF
                        patches = orig_image.unfold(1, 3, 3).unfold(2, size, stride).unfold(3, size, stride)
                        spatial_shape = patches.shape[2:-3]
                        patches = patches.reshape(-1, *patches.shape[-3:])
                        patches = torch.nn.functional.interpolate(patches, size=(224, 224), mode='bicubic')
                        patches = pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(patches)

                        patch_features = model.encode_image(patches)
                        patch_features = patch_features.view(*spatial_shape, *patch_features.shape[1:])

                        np.save(output_path + f"/{scene_path.name}/clip_features/{img_path.name[:-4]}_patch_size_{patch_size:03d}.npy",
                                patch_features.detach().cpu().numpy())

                pass