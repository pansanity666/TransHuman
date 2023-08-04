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
import torchvision

from lib.utils.vis_utils import plotPoints
from lib.utils.SMPL import SMPL
from lib.utils.base_utils import coord_from_blender, pose_spherical
from lib.utils import render_utils

class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()
         
        self.data_root = data_root
        self.split = split  # 'train'
        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()
        self.smpl = SMPL('neutral', './data/smplx/smpl')

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}
        
        human_info = get_human_info.get_human_info(self.split)
        self.human_list = list(human_info.keys())
        self.human_info = human_info
        print('Humans in current dataset are:', self.human_list)

        if self.split == 'test':
            self.human2idx = {}
            for idx, human in enumerate(self.human_list):
                self.human2idx[human] = idx
        
        # used in FrameSampler
        self.human2Nframe_Ncam = {}
        # self.render_w2c = {}
   
        # loop of human ids
        for human in self.human_list:

            data_root = os.path.join(self.data_root, human)
            ann_file = os.path.join(self.data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
            
            self.cams[human] = annots['cams']
            num_cams = len(self.cams[human]['K'])
            
            if cfg.run_mode == 'train':
                target_view = [i for i in range(num_cams)]
            elif cfg.run_mode == 'test':
                target_view = cfg.test.target_view

            print('Human ID: ', human, 'Total camera number:', num_cams, 'Used target cameras: ', target_view)
            assert len(target_view) > 0 
            
            # get ims, cam_inds
            i = human_info[human]['begin_i']
            i_intv = human_info[human]['i_intv']
            ni = human_info[human]['ni']
            
            ### 360 freeviewpoint 
            # _, RT = render_utils.load_cam(ann_file)
            # render_w2c = render_utils.gen_path_virt(RT, render_views=(ni))
            # self.render_w2c[human] = render_w2cs

            ims = np.array([
                np.array(ims_data['ims'])[target_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]
            ])
            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[target_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]
            ])
                
            self.human2Nframe_Ncam[human] = ims.shape
            ims = ims.ravel() # (60, 21) -> 1260
            cam_inds = cam_inds.ravel() # (60, 21) -> 1260
            self.cam_inds.extend(cam_inds)

            ### from relative path to abs path 
            if human in ['CoreView_313', 'CoreView_315']:
                ims = [data_root + '/' + x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' 
                        for x in ims]
            else:
                ims = [data_root + '/' + x for x in ims]

            ### cache start end 
            self.start_end[human] = {}
            self.start_end[human]['start'] = int(ims[0].split('/')[-1][:-4]) # start frame name 
            self.start_end[human]['end'] = int(ims[-1].split('/')[-1][:-4]) # end frame name 
            self.start_end[human]['length'] = self.start_end[human]['end'] - self.start_end[human]['start'] + 1 # 
            self.start_end[human]['intv'] = human_info[human]['i_intv']
            self.ims.extend(ims)
       
        self.nrays = cfg.N_rand
        self.num_humans = len(self.human_list)
        self.epoch = None
    
    def get_mask(self, index):

        data_info = self.ims[index].split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        msk_exist = False
        msk_cihp_exist = False

        msk_path = os.path.join(self.data_root, human, 'mask',
                                camera, frame)[:-4] + '.png'
        msk_exist = os.path.exists(msk_path)
        
        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        msk_path = os.path.join(self.data_root, human, 'mask_cihp',
                                camera, frame)[:-4] + '.png'
        msk_cihp_exist = os.path.exists(msk_path)
        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100
        
        return msk

    def get_input_mask(self, human, index, filename):
        
        msk_exist = False
        msk_cihp_exist = False

        ### get mask path 
        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(self.data_root, human, 'mask',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(self.data_root, human, 'mask',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')
        msk_exist = os.path.exists(msk_path)

        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        ### get mask_cihp path 
        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(self.data_root, human, 'mask_cihp',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(self.data_root, human, 'mask_cihp',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')
        msk_cihp_exist = os.path.exists(msk_path)

        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        return msk

    def get_smpl_vertice(self, human, frame):
        
        if '396' in human:
            vertices_path = os.path.join(self.data_root, human, 'vertices',
                                        '{}.npy'.format(frame))
        else:
            vertices_path = os.path.join(self.data_root, human, cfg.vertices,
                                        '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice

    def prepare_input(self, human, i):
        # load SMPL vertices under world coordinate
        xyz = self.get_smpl_vertice(human, i)
        smpl_vertices = np.array(xyz)
        
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        ### transform smpl from the world coordinate to the smpl coordinate
        # load params for current frame. 
        params = self.get_params(human, i)
        Rh = params['Rh'] # 1, 3 
        Rh = cv2.Rodrigues(Rh)[0].astype(np.float32) # 3, 3 # convert rotation vector to rotation matrix
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, Rh)
        # print(np.dot(xyz, R.T) + Th) # convert from smpl coord to world coord 
        
        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
        smpl_vertice_smplcoord = xyz
        
        # obtain the bounds for coord construction, SMPL coordinate
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz_smpl, max_xyz_smpl = min_xyz, max_xyz

        if cfg.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # get SMPL blend weights from can to obs 
        _, _, blend_mtx = self.smpl(params['poses'], params['shapes'][0])

        return can_bounds, bounds, Rh, Th, center, rot, trans, \
               smpl_vertices, smpl_vertice_smplcoord, min_xyz_smpl, max_xyz_smpl, \
               blend_mtx, params['poses'], params['shapes']
    
    
    def image_to_tensor(self):

        ops = []
        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def color_jitter(self):
        ops = []
        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )
        return transforms.Compose(ops)

    def get_params(self, human, i):
        if '396' in human:
            params_path = os.path.join(self.data_root, human, 'params',
                                    '{}.npy'.format(i))
        else:
            params_path = os.path.join(self.data_root, human, cfg.params,
                                    '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()

        return params
    
    def get_blend_mtx(self, params):
    
        _, _, blend_mtx = self.smpl(params['poses'], params['shapes'][0])
        
        return blend_mtx
    
    def world2smpl(self, xyz, params):
        # xyz: (N_pts, 3)

        Rh = params['Rh'] # 1, 3 
        # convert rotation vector to rotation matrix
        R = cv2.Rodrigues(Rh)[0].astype(np.float32) # 3, 3
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)
        
        return xyz
    
    def get_item(self, index):
        return self.__getitem__(index)
    
    def __getitem__(self, index):
         
        # Somewhat hack here. For keeping the augmentation the same between different runnings. 
        # Thes random seed is determined by training epoch. 
        if self.split == 'train' :
            random.seed(index + self.epoch * cfg.seed)
            
        prob = None
        if self.split == 'train' and cfg.jitter:
            prob = index + self.epoch * cfg.seed
            assert self.epoch is not None, 'Should set epoch before iterating dataloader!'
        
        ### ===================== process target view image =====================
        tar_img_path = self.ims[index]
        data_info = tar_img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]
        cam_ind = self.cam_inds[index] # 13

        # load img 
        img = imageio.imread(tar_img_path) # 0 - 255 
        
        # load mask 
        msk = self.get_mask(index) # loaded msk, 1024x1024, 
        img, msk, target_K, target_R, target_T = self.process_loaded(img, msk, human, cam_ind, prob)

        ### ===================== process input images =====================
        num_inputs = cfg.train_num_views if cfg.run_mode == 'train' else len(cfg.test.input_view)

        # get random input_views (camera views.), somewhat ugly here. 
        if cfg.run_mode == 'train':
            if human in ['CoreView_313', 'CoreView_315']:
                input_view = [i for i in range(len(self.cams[human]['K']))]
                cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22]
            else:
                input_view = [i for i in range(len(self.cams[human]['K']))]
            random.shuffle(input_view)
            input_view = input_view[:num_inputs]
                
        else:
            assert len(cfg.test.input_view)
            if human in ['CoreView_313', 'CoreView_315']:
                cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22]
            input_view = cfg.test.input_view

        input_vizmaps = []
        input_depthmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = [] 
        input_smpl_vertice = [] # len(smpl_vertices) = cfg.time_steps
        input_smpl_vertice_smplcoord = []
        input_img_pths = []
        input_blend_mtx = []
        input_Rh = []
        input_Th = []
        input_poses = []
        input_shapes = [] 
        
        target_frame = frame[:-4] # 0085.jpg -> 0085
        frame_index = int(target_frame)
        zfill = len(target_frame)
        
        # initialize time_mult
        time_mult = self.get_time_mult(human, target_frame)
        # loop of time_steps
        for t in range(cfg.time_steps):
            current_frame = self.get_current_frame(t, human, target_frame, time_mult) 
            # print('current_frame', time_mult, current_frame, t)
            
            current_params = self.get_params(human, current_frame)
            input_blend_mtx.append(self.get_blend_mtx(current_params)) # 6890, 4, 4 
            input_poses.append(current_params['poses'])
            input_shapes.append(current_params['shapes'])
            
            current_smpl = self.get_smpl_vertice(human, current_frame)
            input_smpl_vertice.append(current_smpl)
            input_smpl_vertice_smplcoord.append(self.world2smpl(current_smpl, current_params))
            
            input_Rh.append(cv2.Rodrigues(current_params['Rh'])[0].astype(np.float32))
            input_Th.append(current_params['Th'].astype(np.float32))
            
            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []
            tmp_depthmap = [] 
            tmp_input_img_pths = []
            tmp_input_K = []
            tmp_input_R = []
            tmp_input_T = []
            
            filename = str(current_frame).zfill(zfill) + '.jpg'
            # loop of camera view 
            for i in range(num_inputs):
                idx = input_view[i] # random view index
                
                # load img 
                cam_idx = None
                if human in ['CoreView_313', 'CoreView_315']:
                    cam_idx = cam_idx_list[idx]
                    input_img_path = os.path.join(self.data_root, human, 'Camera (' + str(cam_idx + 1) + ')', filename)
                else:
                    input_img_path = os.path.join(self.data_root, human, 'Camera_B' + str(idx + 1), filename)
    
                tmp_input_img_pths.append(input_img_path)
                input_img = imageio.imread(input_img_path)
                
                # load mask 
                if human in ['CoreView_313', 'CoreView_315']:
                    # NOTE: camera index starts from 1 for 313 and 315 
                    input_msk = self.get_input_mask(human, cam_idx + 1, filename)
                else:
                    input_msk = self.get_input_mask(human, idx + 1, filename)

                # process
                input_img, input_msk, in_K, in_R, in_T = self.process_loaded(input_img, input_msk, human, idx, prob)
                input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

                # load vizmap
                if cfg.rasterize:
                    vizmap_idx = str(current_frame).zfill(zfill)

                    if human in ['CoreView_313', 'CoreView_315']:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera (' + str(
                                                       cam_idx + 1) + ')',
                                                   '{}.npy'.format(vizmap_idx))
                    else:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera_B' + str(idx + 1),
                                                   '{}.npy'.format(vizmap_idx))
                        
                    try: 
                        input_vizmap = np.load(vizmap_path).astype(np.bool)
                    except:
                        print(vizmap_path, 'doesnt exist, using all 1 instead. ')
                        input_vizmap = np.ones((6890)).astype(np.bool)
                    
                    tmp_vizmaps.append(torch.from_numpy(input_vizmap))

                # load depth_map (already 512x512)
                if cfg.depth_map:
                    if human in ['CoreView_313', 'CoreView_315']:
                        depthmap_pth = os.path.join(cfg.depth_root, human,
                                                    'Camera (' + str(
                                                        cam_idx + 1) + ')',
                                                    '{}.pt'.format(vizmap_idx))
                    else:
                        depthmap_pth = os.path.join(cfg.depth_root, human,
                                                    'Camera_B' + str(idx + 1),
                                                    '{}.pt'.format(vizmap_idx))
                    depth_map = torch.load(depthmap_pth)
                    tmp_depthmap.append(depth_map.permute(1,2,0))
                    
                # [-1,1]
                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()

                tmp_imgs.append(input_img)
                tmp_msks.append(input_msk)
                
                tmp_input_K.append(torch.from_numpy(in_K))
                tmp_input_R.append(torch.from_numpy(in_R))
                tmp_input_T.append(torch.from_numpy(in_T))
                
            input_imgs.append(torch.stack(tmp_imgs))
            input_msks.append(torch.stack(tmp_msks))
            if cfg.rasterize:
                input_vizmaps.append(torch.stack(tmp_vizmaps))
            if cfg.depth_map:
                input_depthmaps.append(torch.stack(tmp_depthmap))
            input_img_pths.append(tmp_input_img_pths)

            input_K.append(torch.stack(tmp_input_K))
            input_R.append(torch.stack(tmp_input_R))
            input_T.append(torch.stack(tmp_input_T))           

        i = int(frame[:-4])
        can_bounds, bounds, \
        Rh, Th, center, rot, trans, \
        tar_smpl_vertice, tar_smpl_vertice_smplcoord, \
        min_xyz_smpl, max_xyz_smpl, blend_mtx, poses, shapes = self.prepare_input(human, i)
        
        # K, R, T is the parameters for target view. 
        if self.split == 'train' and  cfg.patch.use_patch_sampling:
            rgb, ray_o, ray_d, near, far, meta = if_nerf_dutils.sample_ray_patch(img, msk, target_K, target_R, target_T, can_bounds, self.nrays, self.split)
            sampled = {
                'rgb': rgb,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
            }
            sampled.update(meta)
            
        else:   
            ### Get dynamic camera
            # cam_ind = i % len(self.render_w2c[human])
            # RT = self.render_w2c[human][cam_ind]
            # target_R, target_T = RT[:3, :3], RT[:3, 3:]
             
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

        ret = {
            'tar_smpl_vertice': tar_smpl_vertice, # this should be the same as input_smpl_vertice when
            'tar_smpl_vertice_smplcoord': tar_smpl_vertice_smplcoord, # smpl under smplcoord for current target time step (used for PE_obs)
            'min_xyz_smpl': min_xyz_smpl,
            'max_xyz_smpl':max_xyz_smpl
        }

        ret.update(sampled)
        
        i = int(os.path.basename(tar_img_path)[:-4]) # frame number 
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human2idx[human]
        
        meta = {
            ### Mics
            'can_bounds': can_bounds, # 3d bounding box under world coord. 
            'bounds': bounds, # 3d bounding box under SMPL coord.
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'cam_ind': cam_ind,
            'frame_index': frame_index, # frame_index in line with image name 
            'human_idx': human_idx, # used for saving results in evaluator
            'human_name': human,
            'time_mult': time_mult,
            ### target meta
            'target_K': target_K,
            'target_R': target_R,
            'target_T': target_T,
            'blend_mtx': blend_mtx, # the blended rotation matrix from canonical to observation SMPL 
            'poses': poses, # poses for target smpl 
            'shapes': shapes, # shapes for target smpl 
            'tar_img_path': tar_img_path,
            'Rh': Rh, # R for target smpl; (3, 3)
            'Th': Th, # Th for target smpl; (1, 3)
            ### input meta
            'input_img_paths': input_img_pths,
            'input_blend_mtx':input_blend_mtx,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_depthmaps': input_depthmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'input_smpl_vertice': input_smpl_vertice, # smpl vertices for all time_step under world coordinate. (used in paint_neural_human )
            'input_smpl_vertice_smplcoord': input_smpl_vertice_smplcoord,
            'input_Rh': input_Rh,
            'input_Th': input_Th,
            'input_poses': input_poses,
            'input_shapes': input_shapes
        }
                
        ret.update(meta)

        return ret

    def get_time_mult(self, human, target_frame):
        
        if self.split == 'train':
            ### original, one current frame + addition temporal frames. 
            raw_mult = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
            random.shuffle(raw_mult)
            raw_mult = raw_mult[:cfg.time_steps - 1]
            if cfg.time_steps > 2:
                raw_mult.sort()
            time_mult = [0]
            time_mult.extend(raw_mult) # print(time_mult) # [0, 3]
            time_mult = np.array(time_mult)
        elif self.split == 'test':
            time_mult = np.array(cfg.time_mult)
            
        return time_mult
    
    def get_current_frame(self, t, human, target_frame, time_mult):
        
        ### get current frame number based on t and time_mult      
        start = self.start_end[human]['start'] # start frame name 
        end = self.start_end[human]['end'] # end frame name 
        intv = self.start_end[human]['intv']
        length = self.start_end[human]['length']
        
        ### get current frame based on time_steps. 
        if self.split == 'train':
            current_frame = ((int(target_frame) + time_mult[t] * intv - start) % length) + start
        elif self.split == 'test':
            current_frame = ((int(target_frame) + time_mult[t] - start) % length) + start
            
        return current_frame
    
    def process_loaded(self, img, msk, human, cam_ind, prob=None):
        ### perform aug + undistort + resize + mask bkgd to loaded img and msk 
        # color aug 
        if self.split == 'train' and cfg.jitter:
            img = Image.fromarray(img)
            torch.manual_seed(prob)
            img = self.jitter(img)
            img = np.array(img)
        img = img.astype(np.float32) / 255.

        # load parameters and undistort 
        K = np.array(self.cams[human]['K'][cam_ind]).astype(np.float32)
        D = np.array(self.cams[human]['D'][cam_ind]).astype(np.float32)
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        R = np.array(self.cams[human]['R'][cam_ind]).astype(np.float32)
        T = (np.array(self.cams[human]['T'][cam_ind]) / 1000.).astype(np.float32) # when generating the annots.py, 1000 is multiplied, so dividing back

        # resize 
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        K[:2] = K[:2] * cfg.ratio
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # mask bkgd
        if cfg.mask_bkgd: # True
            if cfg.white_bkgd: # False 
                img[msk == 0] = 1
            else:
                img[msk == 0] = 0
        
        return img, msk, K.copy(), R.copy(), T.copy() 
    
    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)

    def set_epoch(self, epoch):
        self.epoch = epoch