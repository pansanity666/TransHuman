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
import random

from lib.utils.vis_utils import plotPoints
from lib.utils.SMPL import SMPL

class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        self.smpl = SMPL('neutral', './data/smplx/smpl')
        
        self.split = split
        self.im2tensor = self.image_to_tensor()
        self.cams = {}
        self.ims = []

        self.Ks = {}
        self.Rs = {}
        self.Ts = {}
        self.Ds = {}
        self.start_end = {}
        
        self.cam_names = {} 
        
        self.data_root = data_root
        
        human_info = get_human_info.get_human_info_thu()
        human_list = list(human_info.keys())

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]
            data_root = os.path.join(self.data_root, human)
            ann_file = os.path.join(self.data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
            
            self.cams[human] = annots['cams']
            
            ### 
            # if self.split == 'test':
            #     test_view = [1,5,10,15,20]
            # else:
            #     test_view = list(range(0, num_cams))
            # self.initialized_num_cams = len(test_view)

            i = human_info[human]['begin_i']
            ni = human_info[human]['ni']
            i_intv = human_info[human]['i_intv']
            
            ims = []
            for ims_data in annots['ims'][i:i + ni*i_intv]:
                # print(ims_data)
                _x = []
                for x in np.array(ims_data['ims'])[cfg.test_input_view]:
                    _x.append(data_root + '/' + x)
                ims.append(np.array(_x))
            ims = np.stack(ims)
            # print(ims)
            # print(ims.shape) # 16, 5 

            cam_names = [n.split('/')[0] for n in annots['ims'][0]['ims']]
            self.cam_names[human] = cam_names  

            # ims = [     for ims_data in annots['ims'][i:i + ni*i_intv]]
            ims = np.array(ims)
            
            # ims = np.array([
            #         np.array(
            #             [data_root + '/' + x 
            #         for x in np.array(ims_data['ims']]))[test_view]
            #         for ims_data in annots['ims'][i:i + ni*i_intv]
            #     ]) # (60, 21) -> 1260
            
            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            
            # from relative path to abs path 
            # self.ims[start_idx:start_idx + length] = [
            #         data_root + '/' + x for x in
            #         self.ims[start_idx:start_idx + length]] 
            

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(self.ims[start_idx][0].split('/')[-1][:-4]) # start frame name 
            self.start_end[human]['end'] = int(self.ims[start_idx + length - 1][0].split('/')[-1][:-4]) # end frame name 
            self.start_end[human]['length'] = self.start_end[human]['end'] - self.start_end[human]['start']
            self.start_end[human]['intv'] = human_info[human]['i_intv']


            self.Ks[human] = np.array(self.cams[human]['K'])[
                cfg.test_input_view].astype(
                np.float32)
            self.Rs[human] = np.array(self.cams[human]['R'])[
                cfg.test_input_view].astype(
                np.float32)
            self.Ts[human] = np.array(self.cams[human]['T'])[
                                 cfg.test_input_view].astype(
                np.float32) 
            self.Ds[human] = np.array(self.cams[human]['D'])[
                cfg.test_input_view].astype(
                np.float32)

            # self.ni = cfg.ni

        self.ims = np.array(self.ims)
        self.nrays = cfg.N_rand

    def image_to_tensor(self):

        ops = []
        ops.extend(
            [transforms.ToTensor(), ]
        )
        return transforms.Compose(ops)

    

    def get_smpl_vertice(self, human, frame):

        vertices_path = os.path.join(self.data_root, human, 'new_vertices',
                                     '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice

    def prepare_input(self, human, i):

        vertices_path = os.path.join(self.data_root, human, 'new_vertices',
                                     '{}.npy'.format(i))

        xyz = np.load(vertices_path).astype(np.float32)
        smpl_vertices = None
        if cfg.time_steps == 1:
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

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, human, 'new_params',
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()


        # Rh = params['Rh']  # (1,3)
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # (3,3)
        
        R = params['R'].astype(np.float32)
        Th = params['Th'].astype(np.float32)  # (1,3)


        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        if 'NoSMPLAUG' in cfg.exp_name:
            smpl_vertice_smplcoord = xyz
            xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
        else:
            xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
            smpl_vertice_smplcoord = xyz

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        
        min_xyz_smpl, max_xyz_smpl = min_xyz, max_xyz

        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05


        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        _, _, blended_mtx = self.smpl(params['poses'], params['shapes'])
        
        
        return feature, coord, out_sh, can_bounds, bounds, R, Th, center, rot, trans, \
            smpl_vertices, smpl_vertice_smplcoord, min_xyz_smpl, max_xyz_smpl, blended_mtx, \
                params['poses'], params['shapes']

    
    def get_mask(self, i, nv):

        im = self.ims[i, nv]

        data_info = im.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        msk_path = os.path.join(self.data_root, human, 'mask', camera,
                                frame)[:-4] + '.png'

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

        # msk = (msk | msk_cihp).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        msk = cv2.undistort(msk, self.Ks[human][nv], self.Ds[human][nv])

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)

        return msk
    
    def prepare_inside_pts(self, pts, i):


        human = (self.ims[i][0]).split('/')[-3]

        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)

        for nv in range(self.ims.shape[1]):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            # print(nv, self.Rs[human].shape, self.Ts[human].shape)
            # print(self.Rs[human][nv].shape, self.Ts[human][nv].shape)
            
            RT = np.concatenate([self.Rs[human][nv], self.Ts[human][nv]],
                                axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[human][nv], RT)

            msk = self.get_mask(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    
    def get_input_mask(self, human, camera, frame):

        msk_path = os.path.join(self.data_root, human, 'mask_cihp',
                                camera, frame)[:-4] + '.png'

        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.jpg'
        msk_cihp = imageio.imread(msk_path)

        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        # if not cfg.eval and cfg.erode_edge:
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk


    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        
        # print(index, self.ims[index])
        img_path = self.ims[index][0]

        data_info = img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        i = int(frame[:-4])
        feature, coord, out_sh, can_bounds, bounds, R, Th, \
        center, rot, trans, tmp_smpl_vertices, smpl_vertice_smplcoord, min_xyz_smpl, \
        max_xyz_smpl, blended_mtx, poses, shapes = self.prepare_input(human, i)

        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, index)

        ### --- prepare input images for image feature extraction
        
        num_inputs = cfg.train_num_views if cfg.run_mode == 'train' else len(cfg['test_input_view'])        
        input_view = cfg.test_input_view


        input_vizmaps = []
        input_depthmaps = [] 
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []
        input_img_pths = []
        
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        
        if cfg.time_steps == 1:
            time_mult = [0]
        elif cfg.time_steps > 1:
            if self.split == 'train':
                # random time mult
                raw_mult = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
                random.shuffle(raw_mult)
                raw_mult = raw_mult[:cfg.time_steps - 1]
                if cfg.time_steps > 2:
                    raw_mult.sort()
                time_mult = [0]
                time_mult.extend(raw_mult) # print(time_mult) # [0, 3]
            elif self.split == 'test':
                time_mult = cfg.time_mult

        target_frame = frame[:-4]
 
        frame_index = int(target_frame)
        zfill = len(target_frame)

        for t in range(cfg.time_steps):

            start = self.start_end[human]['start']
            end = self.start_end[human]['end']
            intv = self.start_end[human]['intv']
            length = self.start_end[human]['length']

            # get current frame based on time_steps. 
            if self.split == 'train':
                if t == 0:
                    current_frame = int(target_frame)
                else:
                    current_frame = ((int(target_frame) + time_mult[t] * intv - start) % length) + start
            elif self.split == 'test':
                if t == 0:
                    current_frame = int(target_frame)
                else:
                    current_frame = ((int(target_frame) + time_mult[t] - start) % length) + start

            filename = str(current_frame).zfill(zfill) + '.png'
            if cfg.time_steps > 1:
                smpl_vertices.append(self.get_smpl_vertice(human, current_frame))

            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []
            tmp_depthmap = [] 
            tmp_input_img_pths = []
            
            tmp_input_K = []
            tmp_input_R = []
            tmp_input_T = []
            
            for j in range(len(cfg['test_input_view'])):
                idx = cfg['test_input_view'][j]
                cam_name = self.cam_names[human][idx]
                
                input_img_path = os.path.join(self.data_root, human, cam_name, filename)

                ### load rasterize or depth_map 
                if cfg.rasterize:
                    # fake rasterize for THU 
                    input_vizmap = np.ones((6890)).astype(np.bool)
                    tmp_vizmaps.append(torch.from_numpy(input_vizmap))
                
                # load depth_map (already 512x512)
                if cfg.depth_map:
                    depthmap_pth = os.path.join(cfg.depth_root, human, camera,
                                                    filename.replace('.png', '.pt'))
                    depth_map = torch.load(depthmap_pth)
                    tmp_depthmap.append(depth_map.permute(1,2,0))
                
                input_img = imageio.imread(input_img_path).astype(
                    np.float32) / 255.

                input_msk, _  = self.get_input_mask(human, cam_name, filename)

                in_K = np.array(self.cams[human]['K'][idx]).astype(np.float32)
                in_D = np.array(self.cams[human]['D'][idx]).astype(np.float32)

                input_img = cv2.undistort(input_img, in_K, in_D)
                input_msk = cv2.undistort(input_msk, in_K, in_D)

                in_R = np.array(self.cams[human]['R'][idx]).astype(np.float32)
                in_T = (np.array(self.cams[human]['T'][idx])).astype(np.float32)

                input_img = cv2.resize(input_img, (W, H),
                                       interpolation=cv2.INTER_AREA)
                input_msk = cv2.resize(input_msk, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                
                if cfg.mask_bkgd:
                    if cfg.white_bkgd:
                        input_img[input_msk == 0] = 1
                    else:
                        input_img[input_msk == 0] = 0

                input_msk = (input_msk != 0)
                
                if cfg.use_viz_test and cfg.use_fg_masking:
                    if cfg.ratio == 0.5:
                        border = 5
                    kernel = np.ones((border, border), np.uint8)
                    input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                          kernel)  # [0,255]
                
                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()
                
                in_K[:2] = in_K[:2] * cfg.ratio

                tmp_imgs.append(input_img)
                tmp_msks.append(input_msk)
                
                # if cfg.rasterize:
                #     tmp_vizmaps.append(torch.from_numpy(input_vizmap))
                # if t == 0:
                #     input_K.append(torch.from_numpy(in_K))
                #     input_R.append(torch.from_numpy(in_R))
                #     input_T.append(torch.from_numpy(in_T))

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

        if cfg.time_steps == 1:
            smpl_vertices.append(tmp_smpl_vertices)

            
        ret = {
            'smpl_vertice': smpl_vertices,
            'smpl_vertice_smplcoord': smpl_vertice_smplcoord, # smpl under smplcoord for current target time step (used for PE_obs)
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'min_xyz_smpl': min_xyz_smpl,
            'max_xyz_smpl':max_xyz_smpl,
            'pts': pts,
            'inside': inside,
            'can_bounds': can_bounds
        }

       
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]

        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'frame_index': frame_index, # frame_index in line with image name
            'human_idx': human_idx,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_depthmaps': input_depthmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'blended_mtx': blended_mtx, # the blended rotation matrix from canonical to observation SMPL 
            'poses': poses, # poses for target smpl 
            'shapes': shapes, # shapes for target smpl 
            'img_path': img_path,
            'input_img_paths': input_img_pths
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
