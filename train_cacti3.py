import torch
import numpy as np
import sys, os, time

from nerf import *
import optimize_pose_linear, optimize_pose_all
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt

from metrics import compute_img_metric
import novel_view_test
from load_llff import *


def train():
    parser = config_parser()
    args = parser.parse_args()
    print('spline numbers: ', args.deblur_images)

    imgs_sharp_dir = os.path.join(args.datadir, 'images_test')
    #imgs_sharp = load_imgs(imgs_sharp_dir)

    # Load data images and groundtruth
    K = None
    if args.dataset_type == 'llff':
        

        # load sci measurement and mask
        diffMask = np.load(args.maskdir)
        meas = np.load(args.measdir)
        diffMask = torch.Tensor(diffMask).to(device)
        meas = torch.Tensor(meas).to(device)

        images_all = torch.zeros((diffMask.shape[0], diffMask.shape[1], diffMask.shape[2], 3))
        poses_all = torch.zeros((diffMask.shape[0], 3, 5))
        render_poses = torch.zeros((120, 3, 5))

        hwf = torch.tensor((diffMask.shape[1], diffMask.shape[2], args.f))

        # split train/val/test
        if args.novel_view:
            i_test = torch.arange(0, images_all.shape[0], args.llffhold)
        else:
            i_test = torch.tensor([100]).long()
        i_val = i_test
        i_train = torch.Tensor([i for i in torch.arange(int(images_all.shape[0]))]).long() # all 8 images as training data

        # train data
        images = images_all[i_train]
        
        
        images = meas
        images = images[np.newaxis,:]


        # get poses
        poses_start = poses_all[0]
        #poses_end = poses_start.clone()
        poses_end = poses_all[-1]
        poses_start = poses_start[np.newaxis,:]
        poses_end = poses_end[np.newaxis,:]
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = SE3_to_se3_N(poses_end[:, :3, :4])
        
        poses_org = poses_start.repeat(args.deblur_images, 1, 1)
        poses = poses_org[:, :, :4]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        near = 0.
        far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    print_file = os.path.join(basedir, expname, 'print.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.load_weights:
        if args.linear:
            print('Linear Spline Model Loading!')
            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        else:
            print('Nonlinear Model Loading!')
            model = optimize_pose_all.Model(poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3)
        graph = model.build_network(args)
        optimizer, optimizer_se3 = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)
        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
        global_step = graph_ckpt['global_step']
        

    else:
        if args.linear:
            low, high = 0.0001, 0.005
            
            rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            if focal < 300:
                low_x, high_x = 0.5, 1.0
                rand_x = (high_x - low_x) *torch.rand(1) + low_x
                rand[0,3] = -1 * rand_x
                
            else:
                low_x, high_x = 0.2, 0.5 
                rand_x = (high_x - low_x) *torch.rand(1) + low_x
                rand[0,3] = -1 * rand_x
                
            if args.rotation_perturb:
                low_r1, high_r1 = 0.15, 0.2
                rand_r1 = (high_r1 - low_r1) * torch.rand(1) + low_r1
                rand[0,2] = -1 * rand_r1
                low_r2, high_r2 = 0.35, 0.4
                rand_r2 = (high_r2 - low_r2) * torch.rand(1) + low_r2
                rand[0,1] = -1 * rand_r2
            poses_start_se3 = rand # if pose_end is not identical to pose_start, there is no need to add perturb
            poses_end_se3 = -1 * poses_start_se3.clone()

            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        else:
            
            # Use random initialized values on se3 as the poses
            low, high = 0.001, 0.05
            rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            if focal < 550:
                low_x, high_x = 0.5, 1.0
                rand_x = (high_x - low_x) *torch.rand(1) + low_x
                rand[0,3] = -1 * rand_x
                
            else:
                low_x, high_x = 0.3, 0.5 
                rand_x = (high_x - low_x) *torch.rand(1) + low_x
                rand[0,3] = -1 * rand_x
            
            if args.rotation_perturb:
                low_r1, high_r1 = 0.15, 0.2
                rand_r1 = (high_r1 - low_r1) * torch.rand(1) + low_r1
                rand[0,2] = -1 * rand_r1
                low_r2, high_r2 = 0.3, 0.4
                rand_r2 = (high_r2 - low_r2) * torch.rand(1) + low_r2
                rand[0,1] = -1 * rand_r2
            if args.translation_perturb:
                low_y, high_y = 0.05, 0.1
                rand_y = (high_y - low_y) * torch.rand(1) + low_y
                rand[0,4] = rand_y
            poses_start_se3 = rand 
            poses_end_se3 = -1 * poses_start_se3.clone()
            print("Use random se3 vec to initialize poses......")
        
            poses_se3_1 = (6/7) * poses_start_se3 + (1/7) * poses_end_se3 #poses_1_se3 
            poses_se3_2 = (5/7) * poses_start_se3 + (2/7) * poses_end_se3 #poses_2_se3 
            poses_se3_3 = (4/7) * poses_start_se3 + (3/7) * poses_end_se3 #poses_3_se3 
            poses_se3_4 = (3/7) * poses_start_se3 + (4/7) * poses_end_se3 #poses_4_se3
            poses_se3_5 = (2/7) * poses_start_se3 + (5/7) * poses_end_se3 #poses_5_se3
            poses_se3_6 = (1/7) * poses_start_se3 + (6/7) * poses_end_se3 #poses_6_se3
            print("All poses initializing......")
            model = optimize_pose_all.Model(poses_start_se3, poses_se3_1, poses_se3_2, poses_se3_3, poses_se3_4, poses_se3_5, poses_se3_6, poses_end_se3)

        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3 = model.setup_optimizer(args)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 1

    poses_num = poses.shape[0]

    for i in trange(start, threshold):
    ### core optimization loop ###
        i = i+global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        img_idx = torch.randperm(images.shape[0])
        
        
        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0:
            ret, ray_idx, spline_poses, all_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)
        else:
            ret, ray_idx, spline_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)

        # get image ground truth
        target_s = images[img_idx].reshape(-1, H * W, 3)
        target_s = target_s[:, ray_idx]
        target_s = target_s.reshape(-1, 3)
        diffMask = diffMask.reshape(diffMask.shape[0], H*W)
        mask = diffMask[:, ray_idx]
        target_orig = images_all.reshape(-1, H * W, 3)
        target_orig = target_orig[:, ray_idx]
        target_orig = target_orig.reshape(-1, 3)

        # average
        shape0 = img_idx.shape[0]
        interval = target_s.shape[0] // shape0
        rgb_list = []
        rgb_mat = []
        extras_list = []
        extras_mat = []
        rgb_ = 0
        extras_ = 0

        
        for j in range(0, args.deblur_images):
            rgb_ += torch.multiply(ret['rgb_map'][j * interval:(j + 1) * interval], mask[j].repeat((3, 1)).t())
            rgb_mat.append(ret['rgb_map'][j * interval:(j + 1) * interval])

            if 'rgb0' in ret:
                extras_ += torch.multiply(ret['rgb0'][j * interval:(j + 1) * interval], mask[j].repeat((3,1)).t())
                extras_mat.append(ret['rgb0'][j * interval:(j + 1) * interval])
            if (j + 1)== args.deblur_images:
                rgb_list = rgb_
                if 'rgb0' in ret:
                    extras_list = extras_
        
        
        rgb_blur = rgb_list
        rgb_mat = torch.stack(rgb_mat, 0)
        rgb_mat = rgb_mat.reshape(-1, 3)
        #rgb_blur = rgb_blur.reshape(-1, 3)
        if 'rgb0' in ret:
            extras_blur = extras_list
            extras_mat = torch.stack(extras_mat, 0)
            extras_mat = extras_mat.reshape(-1, 3)
            #extras_blur = extras_blur.reshape(-1, 3)

        # backward
        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        img_loss = img2mse(rgb_blur, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            img_loss0 = img2mse(extras_blur, target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()

        optimizer.step()
        optimizer_se3.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose
        
        


        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item(), psnr0.item()}, rgbmax: {np.max(rgb_blur.detach().cpu().numpy()), np.max(extras_blur.detach().cpu().numpy())}")
            with open(print_file, 'a') as outfile:
                outfile.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}\n")
        if i%100 == 0 and i < 5000:
            print("pose_start_se3: ", spline_poses[0].detach().cpu().numpy(), "pose_end_se3: ", spline_poses[-1].detach().cpu().numpy())
        elif i%1000 == 0:
            print("pose_start_se3: ", spline_poses[0].detach().cpu().numpy(), "pose_end_se3: ", spline_poses[-1].detach().cpu().numpy())

        if i < 10:
            print('coarse_loss:', img_loss0.item())
            with open(print_file, 'a') as outfile:
                outfile.write(f"coarse loss: {img_loss0.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if args.deblur_images % 2 == 0:
                    i_render = torch.arange(i_train.shape[0]) #* (args.deblur_images+1) + args.deblur_images // 2
                else:
                    i_render = torch.arange(i_train.shape[0]) #* args.deblur_images + args.deblur_images // 2
                imgs_render = render_image_test(i, graph, all_poses[i_render], H, W, K, args, need_depth=False)
            
            

        if global_step > args.N_iters + 1:
            break
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
