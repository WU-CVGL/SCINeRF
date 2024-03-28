import torch
import numpy as np
import lpips
import imageio
import cv2
import os, sys, time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

path_gt = '' # Ground truth image path
path_clear = '' # Generated image path

imgfiles = [os.path.join(path_clear, f) for f in sorted(os.listdir(path_clear)) if
            f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]

gtfiles = [os.path.join(path_gt, f) for f in sorted(os.listdir(path_gt)) if
            f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]

def imread(f):
    return imageio.imread(f)

imgs = [imread(f)[..., :3] for f in imgfiles]
imgs_gt = [imread(f)[..., :3] for f in gtfiles]


imgs = np.stack(imgs, 0)
imgs_gt = np.stack(imgs_gt, 0)

lpips_model = lpips.LPIPS(net="alex")

lpips1_all = []
lpips2_all = []

for i in range(imgs.shape[0]):
    image_gt = imgs_gt[i]
    image_clear = imgs[i]
    image_gt = torch.tensor(np.array(image_gt)).permute(2, 0, 1).unsqueeze(0).float() / 255
    image_clear = torch.tensor(np.array(image_clear)).permute(2, 0, 1).unsqueeze(0).float() / 255


    lpips2 = lpips_model(image_gt, image_clear)
    
    lpips2_all.append(lpips2.detach().cpu().numpy())
    print("LPIPS is ", lpips2.item())


lpips_clear = np.mean(lpips2_all)
print('Mean LPIPS of clear is ',lpips_clear)
np.save(os.path.join(path_clear,'lpips.npy'), lpips2_all)


