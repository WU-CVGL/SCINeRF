# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image

**[Paper](https://arxiv.org/abs/2403.20018)**

**[Yunhao Li](https://yunhaoli2020.github.io/), Xiaodong Wang, Ping Wang, [Xin Yuan](https://sites.google.com/site/eiexyuan/) and [Peidong Liu](https://ethliup.github.io/) [CVPR 2024 Highlight]**

![teaser2](./doc/fig1_7(2).jpg)

This is an official PyTorch implementation of the paper SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image (CVPR 2024).

SCINeRF retrieves the 3D scenes, together with compressed images, from a single temporal-compressed snapshot compressive image.


## ✨News


⚡ **[2024.02]** Our paper has been accepted by CVPR 2024!

⚡ **[2024.04]** Our paper has been selected as **CVPR 2024 Highlight** (~2.8% acceptance rate)!

## Novel View Synthesis
<img src="https://github.com/WU-CVGL/SCINeRF/blob/main/doc/airplants_25_spiral_200000_rgb-ezgif.com-video-to-gif-converter.gif" width="30%"> <img src="https://github.com/WU-CVGL/SCINeRF/blob/main/doc/factory_all_spiral_200000_rgb-ezgif.com-video-to-gif-converter.gif" width="45%">




## Image Restoration Results
![results](./doc/results_github.jpg)
## Method overview
![method](./doc/fig2_2.jpg)
When capturing the scene, the snapshot compressive imaging (SCI) camera moves alongside a trajectoy and capture the scene into an SCI measurement.
We follow the real physical image formation process of snapshot compressive imaging (SCI) to synthesize SCI measurement from NeRF. Both NeRF and the motion trajectories are estimated by maximizing the photometric consistency between the synthesized sci measurement and the real measurement.


## Quickstart

### 1. Setup environment

```
git clone git@github.com:WU-CVGL/SCINeRF.git
cd SCINeRF
pip install -r requirements.txt
```

### 2. Download datasets

You can download the data [here](https://drive.google.com/file/d/19cR54NiB92-GjPQLm7DiojN1eTy7ECYE/view?usp=sharing).

Each scene contains an SCI measurement in 'meas.npy', and corresponding masks in 'mask.npy' (in some scenes they are 'mask_25.npy" and 'meas_25.npy').

### 3. Configs

Change the data path and other parameters (if needed) in `configs/cozy2room.txt`. We use *cozy2room* scene as an example.

### 4. Demo with our pre-trained model

You can test our code and render retrieved images from compressed image with the provided checkpoints. To do this, you should first download the checkpoints [here](https://drive.google.com/file/d/1Ko6rNwcatG7RMAVRnHyPVj4EdfzIiwRM/view?usp=sharing), then put the weight file under the corresponding logs folder `./logs/cozy2room`, and then change the parameter `load_weights=True` in `cozy2room.txt`, finally run

```
python train_cacti3.py --config configs/cozy2room.txt
```

### 5. Training

```
python train_cacti3.py --config configs/cozy2room.txt
```

After training, you can get retrieved images from SCI measurement.

## Your own data

`mask.npy`: This file is the binary masks you used during the SCI imaging process.

`meas.npy`: This file is the SCI measurement, i.e., the compressed image.

Put two files under a folder at  `./data`. Then create a new config file in the `./configs` folder, make sure that the data directory is correct. Finally, modify he h, w, and f parameters in your config file, which represents image height, width and focal length of the camera. 

## Citation

If you find this useful, please consider citing our paper:

```bibtex
@InProceedings{li2024scinerf,
    author    = {Li, Yunhao and Wang, Xiaodong and Wang, Ping and Yuan, Xin and Liu, Peidong},
    title     = {{SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image}},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```


## Acknowledgment

The overall framework, metrics computing and camera transformation are derived from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/) and [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF) respectively. We appreciate the effort of the contributors to these repositories.
