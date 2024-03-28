import torch.nn

import Spline
import nerf


class Model(nerf.Model):
    def __init__(self, se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7):
        super().__init__()
        self.se3_0 = se3_0
        self.se3_1 = se3_1
        self.se3_2 = se3_2
        self.se3_3 = se3_3
        self.se3_4 = se3_4
        self.se3_5 = se3_5
        self.se3_6 = se3_6
        self.se3_7 = se3_7

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.se3_0.shape[0], 6*4)

        start_end = torch.cat([self.se3_0, self.se3_1, self.se3_2, self.se3_3, self.se3_4, self.se3_5, self.se3_6, self.se3_7], -1)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:][img_idx]

        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, args.deblur_images)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        se3_4 = se3_4[seg_pos_x, :]
        se3_5 = se3_5[seg_pos_x, :]
        se3_6 = se3_6[seg_pos_x, :]
        se3_7 = se3_7[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7], 0)

        #spline_poses_test = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, torch.tensor([0]), 2)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num+1
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:][img_idx]

        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, deblur_images_num)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7], 0)
        #spline_poses = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, pose_nums, deblur_images_num)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
    


class ModelHCR(nerf.Model):
    def __init__(self, se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15):
        super().__init__()
        self.se3_0 = se3_0
        self.se3_1 = se3_1
        self.se3_2 = se3_2
        self.se3_3 = se3_3
        self.se3_4 = se3_4
        self.se3_5 = se3_5
        self.se3_6 = se3_6
        self.se3_7 = se3_7
        self.se3_8 = se3_8
        self.se3_9 = se3_9
        self.se3_10 = se3_10
        self.se3_11 = se3_11
        self.se3_12 = se3_12
        self.se3_13 = se3_13
        self.se3_14 = se3_14
        self.se3_15 = se3_15

    def build_network(self, args):
        self.graph = GraphHCR(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.se3_0.shape[0], 6*4)

        start_end = torch.cat([self.se3_0, self.se3_1, self.se3_2, self.se3_3, self.se3_4, self.se3_5, self.se3_6, self.se3_7, self.se3_8, self.se3_9, self.se3_10, self.se3_11, self.se3_12, self.se3_13, self.se3_14, self.se3_15], -1)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class GraphHCR(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]

        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, args.deblur_images)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        se3_4 = se3_4[seg_pos_x, :]
        se3_5 = se3_5[seg_pos_x, :]
        se3_6 = se3_6[seg_pos_x, :]
        se3_7 = se3_7[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15], 0)

        #spline_poses_test = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, torch.tensor([0]), 2)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num+1
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]

        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, deblur_images_num)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15], 0)
        #spline_poses = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, pose_nums, deblur_images_num)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
    

class ModelHCR24(nerf.Model):
    def __init__(self, se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23):
        super().__init__()
        self.se3_0 = se3_0
        self.se3_1 = se3_1
        self.se3_2 = se3_2
        self.se3_3 = se3_3
        self.se3_4 = se3_4
        self.se3_5 = se3_5
        self.se3_6 = se3_6
        self.se3_7 = se3_7
        self.se3_8 = se3_8
        self.se3_9 = se3_9
        self.se3_10 = se3_10
        self.se3_11 = se3_11
        self.se3_12 = se3_12
        self.se3_13 = se3_13
        self.se3_14 = se3_14
        self.se3_15 = se3_15
        self.se3_16 = se3_16
        self.se3_17 = se3_17
        self.se3_18 = se3_18
        self.se3_19 = se3_19
        self.se3_20 = se3_20
        self.se3_21 = se3_21
        self.se3_22 = se3_22
        self.se3_23 = se3_23

    def build_network(self, args):
        self.graph = GraphHCR24(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.se3_0.shape[0], 6*4)

        start_end = torch.cat([self.se3_0, self.se3_1, self.se3_2, self.se3_3, self.se3_4, self.se3_5, self.se3_6, self.se3_7, self.se3_8, self.se3_9, self.se3_10, self.se3_11, self.se3_12, self.se3_13, self.se3_14, self.se3_15, self.se3_16, self.se3_17, self.se3_18, self.se3_19, self.se3_20, self.se3_21, self.se3_22, self.se3_23], -1)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class GraphHCR24(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]
        se3_16 = self.se3.weight[:, 96:102][img_idx]
        se3_17 = self.se3.weight[:, 102:108][img_idx]
        se3_18 = self.se3.weight[:, 108:114][img_idx]
        se3_19 = self.se3.weight[:, 114:120][img_idx]
        se3_20 = self.se3.weight[:, 120:126][img_idx]
        se3_21 = self.se3.weight[:, 126:132][img_idx]
        se3_22 = self.se3.weight[:, 132:138][img_idx]
        se3_23 = self.se3.weight[:, 138:144][img_idx]

        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, args.deblur_images)
        
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23], 0)

        #spline_poses_test = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, torch.tensor([0]), 2)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num+1
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]
        se3_16 = self.se3.weight[:, 96:102][img_idx]
        se3_17 = self.se3.weight[:, 102:108][img_idx]
        se3_18 = self.se3.weight[:, 108:114][img_idx]
        se3_19 = self.se3.weight[:, 114:120][img_idx]
        se3_20 = self.se3.weight[:, 120:126][img_idx]
        se3_21 = self.se3.weight[:, 126:132][img_idx]
        se3_22 = self.se3.weight[:, 132:138][img_idx]
        se3_23 = self.se3.weight[:, 138:144][img_idx]

        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, deblur_images_num)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23], 0)
        #spline_poses = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, pose_nums, deblur_images_num)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses

class ModelHCR32(nerf.Model):
    def __init__(self, se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23, se3_24, se3_25, se3_26, se3_27, se3_28, se3_29, se3_30, se3_31):
        super().__init__()
        self.se3_0 = se3_0
        self.se3_1 = se3_1
        self.se3_2 = se3_2
        self.se3_3 = se3_3
        self.se3_4 = se3_4
        self.se3_5 = se3_5
        self.se3_6 = se3_6
        self.se3_7 = se3_7
        self.se3_8 = se3_8
        self.se3_9 = se3_9
        self.se3_10 = se3_10
        self.se3_11 = se3_11
        self.se3_12 = se3_12
        self.se3_13 = se3_13
        self.se3_14 = se3_14
        self.se3_15 = se3_15
        self.se3_16 = se3_16
        self.se3_17 = se3_17
        self.se3_18 = se3_18
        self.se3_19 = se3_19
        self.se3_20 = se3_20
        self.se3_21 = se3_21
        self.se3_22 = se3_22
        self.se3_23 = se3_23
        self.se3_24 = se3_24
        self.se3_25 = se3_25
        self.se3_26 = se3_26
        self.se3_27 = se3_27
        self.se3_28 = se3_28
        self.se3_29 = se3_29
        self.se3_30 = se3_30
        self.se3_31 = se3_31

    def build_network(self, args):
        self.graph = GraphHCR32(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.se3_0.shape[0], 6*4)

        start_end = torch.cat([self.se3_0, self.se3_1, self.se3_2, self.se3_3, self.se3_4, self.se3_5, self.se3_6, self.se3_7, self.se3_8, self.se3_9, self.se3_10, self.se3_11, self.se3_12, self.se3_13, self.se3_14, self.se3_15, self.se3_16, self.se3_17, self.se3_18, self.se3_19, self.se3_20, self.se3_21, self.se3_22, self.se3_23, self.se3_24, self.se3_25, self.se3_26, self.se3_27, self.se3_28, self.se3_29, self.se3_30, self.se3_31], -1)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class GraphHCR32(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]
        se3_16 = self.se3.weight[:, 96:102][img_idx]
        se3_17 = self.se3.weight[:, 102:108][img_idx]
        se3_18 = self.se3.weight[:, 108:114][img_idx]
        se3_19 = self.se3.weight[:, 114:120][img_idx]
        se3_20 = self.se3.weight[:, 120:126][img_idx]
        se3_21 = self.se3.weight[:, 126:132][img_idx]
        se3_22 = self.se3.weight[:, 132:138][img_idx]
        se3_23 = self.se3.weight[:, 138:144][img_idx]
        se3_24 = self.se3.weight[:, 144:150][img_idx]
        se3_25 = self.se3.weight[:, 150:156][img_idx]
        se3_26 = self.se3.weight[:, 156:162][img_idx]
        se3_27 = self.se3.weight[:, 162:168][img_idx]
        se3_28 = self.se3.weight[:, 168:174][img_idx]
        se3_29 = self.se3.weight[:, 174:180][img_idx]
        se3_30 = self.se3.weight[:, 180:186][img_idx]
        se3_31 = self.se3.weight[:, 186:192][img_idx]

        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, args.deblur_images)
        
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23, se3_24, se3_25, se3_26, se3_27, se3_28, se3_29, se3_30, se3_31], 0)

        #spline_poses_test = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, torch.tensor([0]), 2)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num+1
        se3_0 = self.se3.weight[:, :6][img_idx]
        se3_1 = self.se3.weight[:, 6:12][img_idx]
        se3_2 = self.se3.weight[:, 12:18][img_idx]
        se3_3 = self.se3.weight[:, 18:24][img_idx]
        se3_4 = self.se3.weight[:, 24:30][img_idx]
        se3_5 = self.se3.weight[:, 30:36][img_idx]
        se3_6 = self.se3.weight[:, 36:42][img_idx]
        se3_7 = self.se3.weight[:, 42:48][img_idx]
        se3_8 = self.se3.weight[:, 48:54][img_idx]
        se3_9 = self.se3.weight[:, 54:60][img_idx]
        se3_10 = self.se3.weight[:, 60:66][img_idx]
        se3_11 = self.se3.weight[:, 66:72][img_idx]
        se3_12 = self.se3.weight[:, 72:78][img_idx]
        se3_13 = self.se3.weight[:, 78:84][img_idx]
        se3_14 = self.se3.weight[:, 84:90][img_idx]
        se3_15 = self.se3.weight[:, 90:96][img_idx]
        se3_16 = self.se3.weight[:, 96:102][img_idx]
        se3_17 = self.se3.weight[:, 102:108][img_idx]
        se3_18 = self.se3.weight[:, 108:114][img_idx]
        se3_19 = self.se3.weight[:, 114:120][img_idx]
        se3_20 = self.se3.weight[:, 120:126][img_idx]
        se3_21 = self.se3.weight[:, 126:132][img_idx]
        se3_22 = self.se3.weight[:, 132:138][img_idx]
        se3_23 = self.se3.weight[:, 138:144][img_idx]
        se3_24 = self.se3.weight[:, 144:150][img_idx]
        se3_25 = self.se3.weight[:, 150:156][img_idx]
        se3_26 = self.se3.weight[:, 156:162][img_idx]
        se3_27 = self.se3.weight[:, 162:168][img_idx]
        se3_28 = self.se3.weight[:, 168:174][img_idx]
        se3_29 = self.se3.weight[:, 174:180][img_idx]
        se3_30 = self.se3.weight[:, 180:186][img_idx]
        se3_31 = self.se3.weight[:, 186:192][img_idx]

        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_0.shape[0], 1)
        seg_pos_x = torch.arange(se3_0.shape[0]).reshape([se3_0.shape[0], 1]).repeat(1, deblur_images_num)
        '''
        se3_0 = se3_0[seg_pos_x, :]
        se3_1 = se3_1[seg_pos_x, :]
        se3_2 = se3_2[seg_pos_x, :]
        se3_3 = se3_3[seg_pos_x, :]
        '''
        spline_poses_se3 = torch.cat([se3_0, se3_1, se3_2, se3_3, se3_4, se3_5, se3_6, se3_7, se3_8, se3_9, se3_10, se3_11, se3_12, se3_13, se3_14, se3_15, se3_16, se3_17, se3_18, se3_19, se3_20, se3_21, se3_22, se3_23, se3_24, se3_25, se3_26, se3_27, se3_28, se3_29, se3_30, se3_31], 0)
        #spline_poses = Spline.SplineN_cubic(se3_0, se3_1, se3_2, se3_3, pose_nums, deblur_images_num)
        spline_poses = Spline.se3_to_SE3_N(spline_poses_se3)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
