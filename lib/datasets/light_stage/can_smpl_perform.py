import torch
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.datasets import get_human_info
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import random
import pdb
from lib.datasets.light_stage.can_smpl import Dataset as Base_Dataset


### Inherit mostly from base daaset, except for dynamic rotation of target camera.
class Dataset(Base_Dataset):
    def __init__(self, *args, **kwargs):

        super(Dataset, self).__init__(*args, **kwargs)
        
        self.render_w2c = {}
        # loop of human ids
        for human in self.human_list:
            ann_file = os.path.join(self.data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
 
            # get ims, cam_inds
            i = self.human_info[human]['begin_i']
            i_intv = self.human_info[human]['i_intv']
            ni = self.human_info[human]['ni']
            
            _, RT = render_utils.load_cam(ann_file)
            render_w2c = render_utils.gen_path_virt(RT, render_views=(ni))
            self.render_w2c[human] = render_w2c
            
    def __getitem__(self, index):
         
        ret = Base_Dataset.__getitem__(self, index)
        
        ### update the target ray info with dynamic cameras ### 
        tar_img_path = self.ims[index]
        data_info = tar_img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]
        cam_ind = self.cam_inds[index] 

        # load img 
        img = imageio.imread(tar_img_path) # 0 - 255 
        
        # load mask 
        msk = self.get_mask(index) # loaded msk, 1024x1024, 
        img, msk, target_K, target_R, target_T = self.process_loaded(img, msk, human, cam_ind, prob=None)

        i = int(frame[:-4])
        can_bounds, bounds, \
        Rh, Th, center, rot, trans, \
        tar_smpl_vertice, tar_smpl_vertice_smplcoord, \
        min_xyz_smpl, max_xyz_smpl, blend_mtx, poses, shapes = self.prepare_input(human, i)
        
        # get dynamic camera 
        cam_ind = i % len(self.render_w2c[human])
        RT = self.render_w2c[human][cam_ind]
        target_R, target_T = RT[:3, :3], RT[:3, 3:]
            
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(img, msk, target_K, target_R, target_T, can_bounds, self.nrays, self.split)
        acc = if_nerf_dutils.get_acc(coord_, msk)
        
        sampled = {
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc, # what for ?
            'mask_at_box': mask_at_box, # what for ? L1 loss only supervise for the foreground pixel? 
        }

        ret.update(sampled)

        return ret
    