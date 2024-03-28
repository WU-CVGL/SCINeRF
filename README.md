# SCINeRF
[CVPR2024] SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image


This is an official PyTorch implementation of the paper SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image (CVPR 2024). Authors: [Yunhao Li](https://yunhaoli2020.github.io/), Xiaodong Wang, Ping Wang, [Xin Yuan](https://sites.google.com/site/eiexyuan/) and [Peidong Liu](https://ethliup.github.io/).

SCINeRF retrieves the 3D scenes, together with compressed images, from a single temporal-compressed snapshot compressive image.

Codes and Data will be available soon!

## ✨News


⚡ **[2024.02]** Our paper has been accepted by CVPR 2024!

## Novel View Synthesis

## Image Restoration Result




## Method overview

When capturing the scene, the snapshot compressive imaging (SCI) camera moves alongside a trajectoy and capture the scene into an SCI measurement.
We follow the real physical image formation process of snapshot compressive imaging (SCI) to synthesize SCI measurement from NeRF. Both NeRF and the motion trajectories are estimated by maximizing the photometric consistency between the synthesized sci measurement and the real measurement.




## Acknowledgment

The overall framework, metrics computing and camera transformation are derived from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/) and [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF) respectively. We appreciate the effort of the contributors to these repositories.
