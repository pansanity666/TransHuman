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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import pdb
import random
import time

from lib.utils.vis_utils import plotPoints
from lib.utils.SMPL import SMPL
from lib.datasets.light_stage.can_smpl import Dataset as Base_Dataset

class Dataset(Base_Dataset):
    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)


    def prepare_inside_pts(self, pts, i):

        human = (self.ims[i]).split('/')[-3]

        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)

        # for nv in range(self.ims.shape[1]):
        #     ind = inside == 1
        #     pts3d_ = pts3d[ind]

        #     RT = np.concatenate([self.Rs[human][nv], self.Ts[human][nv]],
        #                         axis=1)
        #     pts2d = base_utils.project(pts3d_, self.Ks[human][nv], RT)

        #     msk = self.get_mask(i)
        #     H, W = msk.shape
        #     pts2d = np.round(pts2d).astype(np.int32)
        #     pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        #     pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
        #     msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

        #     inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside


    def __getitem__(self, index):
        
        ret = Base_Dataset.__getitem__(self, index)

        ### get mesh inference pts 
        tar_img_path = self.ims[index]
        data_info = tar_img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        i = int(frame[:-4])
        can_bounds, bounds, \
        Rh, Th, center, rot, trans, \
        tar_smpl_vertice, tar_smpl_vertice_smplcoord, \
        min_xyz_smpl, max_xyz_smpl, blend_mtx, poses, shapes = self.prepare_input(human, i)
        
        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        # inside = self.prepare_inside_pts(pts, index) ## inside mask 
   
        mesh_ret = {
            'pts': pts,
            # 'inside': inside,
        }

        ret.update(mesh_ret)
            
        return ret
