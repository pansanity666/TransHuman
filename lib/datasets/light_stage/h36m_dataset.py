import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData

from lib.datasets import get_human_info
from torchvision import transforms
import torch


from lib.utils.vis_utils import plotPoints
from lib.utils.SMPL import SMPL

class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        # self.human = human
        self.split = split
        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()
        self.smpl = SMPL('neutral', './data/smplx/smpl')

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {} 
        self.cam_names = {}

        human_info = get_human_info.get_human_info_h36m()
        human_list = list(human_info.keys())
        self.human_info = human_info

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        self.human2Nframe_Ncam = {}
        
        for human in human_list:
            
            data_root = os.path.join(self.data_root, human)
            ann_file = os.path.join(self.data_root, human, 'Posing', 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()

            self.cams[human] = annots['cams']
            num_cams = len(self.cams[human]['K'])

            # H36M only supports test mode ATM. 
            assert self.split == 'test'
            target_view = cfg.test_target_view
      
            i = human_info[human]['begin_i']
            i_intv = human_info[human]['i_intv']
            ni = human_info[human]['ni']
            
            ims = np.array([
                np.array(ims_data['ims'])[target_view]
                for ims_data in annots['ims'][i:i + ni*i_intv][::i_intv]
            ])

            self.human2Nframe_Ncam[human] = ims.shape
            ims = ims.ravel() # (60, 21) -> 1260
            
            # cache cam_name for each human 
            cam_names = [n.split('/')[0] for n in annots['ims'][0]['ims']]
            self.cam_names[human] = cam_names
            
            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[target_view]
                for ims_data in annots['ims'][i:i + ni*i_intv][::i_intv]
            ]).ravel() # (60, 21) -> 1260

            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)

            # from relative path to abs path 
            self.ims[start_idx:start_idx + length] = [
                    data_root + '/Posing/' + x for x in
                    self.ims[start_idx:start_idx + length]] 

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(self.ims[start_idx].split('/')[-1][:-4]) # start frame name 
            self.start_end[human]['end'] = int(self.ims[start_idx + length - 1].split('/')[-1][:-4]) # end frame name 
            self.start_end[human]['length'] = self.start_end[human]['end'] - self.start_end[human]['start']
            self.start_end[human]['intv'] = human_info[human]['i_intv']

        self.nrays = cfg.N_rand
        self.epoch = None
        self.num_humans = len(human_list)
    
    def color_jitter(self):
        
        if 'NOJIT' in cfg.exp_name:
            return transforms.Compose([])
            
        ops = []

        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )

        if 'GRAYAUG' in cfg.exp_name:
            ops.extend([transforms.RandomGrayscale(p=0.2)])
        
        return transforms.Compose(ops)

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def get_mask(self, human, camera, frame):

        msk_path = os.path.join(self.data_root, human, 'Posing', 'mask_cihp',
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


    def prepare_input(self, human, i):
        # load SMPL vertices under world coordinate
        # if '396' in human:
        #     vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
        #                                 '{}.npy'.format(i))
        # else:
        #     vertices_path = os.path.join(cfg.virt_data_root, human, cfg.vertices,
        #                                 '{}.npy'.format(i))

        vertices_path = os.path.join(self.data_root, human, 'Posing', 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        smpl_vertices = None
        if cfg.time_steps == 1:
            smpl_vertices = np.array(xyz)

        # what for ?
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
        # if '396' in human:
        #     params_path = os.path.join(cfg.virt_data_root, human, 'params',
        #                             '{}.npy'.format(i))
        # else:
        #     params_path = os.path.join(cfg.virt_data_root, human, cfg.params,
        #                             '{}.npy'.format(i))

        params_path = os.path.join(self.data_root, human, 'Posing', 'new_params', '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        # print(params.keys()) # dict_keys(['poses', 'Rh', 'Th', 'shapes'])

        # print(xyz.shape) # 6890, 3 
        Rh = params['Rh'] # 1, 3 
        # convert rotation vector to rotation matrix
        R = cv2.Rodrigues(Rh)[0].astype(np.float32) # 3, 3
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)
        # print(Rh.shape) # 1, 3 
        # print(Th.shape) # 1, 3 
        # print(xyz.shape) # 6890, 3 
        # print(np.dot(xyz, R.T) + Th) # convert from smpl coord to world coord 
        
        # transformation augmentation
        if 'NoSMPLAUG' in cfg.exp_name:
            smpl_vertice_smplcoord = xyz
            xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
        else:
            xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
            smpl_vertice_smplcoord = xyz

        # smpl_vertice_smplcoord = xyz
        # xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz) 
            
        ## obtain the bounds for coord construction, SMPL coordinate
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
       
        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32) # what for ? # 6890, 6

        ## construct the coordinate and out_sh
        # dhw = zyx
        # xyz: vertex under SMPL coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]

        voxel_size = np.array(cfg.voxel_size)

        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32) # (6890, 3), voxelized coordinate for SMPL vertex. 

        # print(max_dhw - min_dhw) # [0.8509866 1.7096167 1.4288788]
        # print(voxel_size) # [0.005 0.005 0.005]

        ##### TODO

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32) #[180 342 292]
        x = 32 
        out_sh = (out_sh | (x - 1)) + 1 # [192 352 320]

        # get SMPL blend weights from can to obs 
        _, _, blended_mtx = self.smpl(params['poses'], params['shapes'][0])
        # print(T.shape) # 6890, 4, 4

        # blended_mtx = torch.inverse(torch.tensor(blended_mtx[..., :3, :3]))
        # print(blended_mtx.shape)
        # assert False

        # print(params['poses'].shape) # (1, 72)
        # print(params['shapes'].shape) # (1, 10)

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vertices, smpl_vertice_smplcoord, min_xyz_smpl, max_xyz_smpl, blended_mtx, params['poses'], params['shapes']


    def get_smpl_vertice(self, human, frame):
        
        vertices_path = os.path.join(self.data_root, human, 'Posing', 'new_vertices',
                                        '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice


    def load_img_msk(self, human, camera, frame, cam_ind):
        # load and resize 
        img_path = os.path.join(self.data_root, human, 'Posing', camera, frame)
        img = imageio.imread(img_path).astype(np.float32) # RGB 
        msk, orig_msk = self.get_mask(human, camera, frame)

        # print(img.shape) # (1000, 1000, 3)
        if self.split == 'train': 
            if 'RANDBG' in cfg.exp_name:
                bgcolor = (np.random.rand(3) * 255.).astype('float32')  # 0 - 255 
                img = self.paint_background(msk, img, bgcolor)
            elif cfg.jitter: #defaul yes 
                img = Image.fromarray(img)
                # if dist.get_rank() == 0: print(prob)
                torch.manual_seed(prob)
                img = self.jitter(img)
                img = np.array(img)
        img = img.astype(np.float32) / 255.
    
        K = np.array(self.cams[human]['K'][cam_ind])
        D = np.array(self.cams[human]['D'][cam_ind])

        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams[human]['R'][cam_ind])
        # when generating the annots.py, 1000 is multiplied, so dividing back
        T = np.array(self.cams[human]['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.split == 'train' and 'RANDBG' in cfg.exp_name:
            pass 
        else:
            # use original balck mask during test time
            if cfg.mask_bkgd:
                if cfg.white_bkgd:
                    img[msk == 0] = 1
                else:
                    img[msk == 0] = 0

        K[:2] = K[:2] * cfg.ratio

        target_K = K.copy() # intrinsics
        target_R = R.copy() # rotation 
        target_T = T.copy() # transformation

        return img, msk, target_K, target_R, target_T, H, W 
    
    
    def image_to_tensor(self):

        ops = []

        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def __getitem__(self, index):

        ### fix seed 
        if self.split == 'train' :
            random.seed(index + self.epoch * cfg.seed)
            # np.random.seed(index + self.epoch * cfg.seed)
            # torch.cuda.manual_seed_all(index + self.epoch * cfg.seed)
        if self.split == 'train' and cfg.jitter:
            prob = index + self.epoch * cfg.seed
            assert self.epoch is not None, 'Should set epoch before iterating dataloader!'
        
        ### init meta 
        img_path = self.ims[index].split('/')
        human = img_path[-4]
        camera = img_path[-2]
        frame = img_path[-1]
        cam_ind = self.cam_inds[index]

        ### load and resize 
        img, msk, target_K, target_R, target_T, H, W  = self.load_img_msk(human, camera, frame, cam_ind)
        
        ### get input view
        # get random input_views (camera views.)
        if cfg.run_mode == 'train':
            num_inputs = cfg.train_num_views
            if cfg.mono_input:
                assert num_inputs == 1
                input_view = [cam_ind]
            else:
                input_view = [i for i in range(len(self.cams[human]['K']))]
                random.shuffle(input_view)
                input_view = input_view[:num_inputs]
        else:
            assert len(cfg.test_input_view)
            input_view = cfg.test_input_view
            num_inputs = len(input_view)

        ### temporal 
        input_vizmaps = []
        input_depthmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = [] 
        smpl_vertices = [] # len(smpl_vertices) = cfg.time_steps
        input_img_pths = []

        ## init time mult
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

        # print(frame) # 0085.jpg
        target_frame = frame[:-4]
        # print(target_frame) # 0085
        frame_index = int(target_frame)
        zfill = len(target_frame)

        # loop of time_steps
        for t in range(cfg.time_steps):

            start = self.start_end[human]['start'] # start frame name 
            end = self.start_end[human]['end'] # end frame name 
            intv = self.start_end[human]['intv']
            length = self.start_end[human]['length']
            # print(start, end) # 1, 60
            
            # get current frame based on time_steps. 
            if self.split == 'train':
                if t == 0:
                    current_frame = frame_index
                else:
                    current_frame = ((frame_index + time_mult[t] * intv - start) % length) + start
            elif self.split == 'test':
                if t == 0:
                    current_frame = frame_index
                else:
                    current_frame = ((frame_index + time_mult[t] - start) % length) + start
                    
            filename = str(current_frame).zfill(zfill) + '.jpg'

            if cfg.time_steps > 1:
                # if time steps > 1, new smpls shoud be loaded. 
                smpl_vertices.append(self.get_smpl_vertice(human, current_frame))

            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []
            tmp_depthmap = [] 
            tmp_input_img_pths = []

            tmp_input_K = []
            tmp_input_R = []
            tmp_input_T = []
            
            # loop of camera view 
            for i in range(num_inputs):

                idx = input_view[i] # random view index
                cam_name = self.cam_names[human][idx]

                input_img_path = os.path.join(self.data_root, human, 'Posing', cam_name, filename)
                tmp_input_img_pths.append(input_img_path)
                
                # load mask 
                input_msk, _ = self.get_mask(human, cam_name, filename)

                # color jitter
                ## load and process image 
                input_img = imageio.imread(input_img_path)
                if self.split == 'train':
                    if 'RANDBG' in cfg.exp_name:
                        input_img = self.paint_background(input_msk, input_img, bgcolor) # 0-255, H,W,3
                    elif cfg.jitter:    
                        input_img = Image.fromarray(input_img)
                        torch.manual_seed(prob)
                        input_img = self.jitter(input_img)
                        input_img = np.array(input_img)
                input_img = input_img.astype(np.float32) / 255. 
                
                # load vizmap
                if cfg.rasterize:
                    # vizmap_idx = str(current_frame).zfill(zfill)
                    # if human in ['CoreView_313', 'CoreView_315']:
                    #     vizmap_path = os.path.join(cfg.rasterize_root, human,
                    #                                'visibility',
                    #                                'Camera (' + str(
                    #                                    cam_idx + 1) + ')',
                    #                                '{}.npy'.format(vizmap_idx))
                    # else:
                    #     vizmap_path = os.path.join(cfg.rasterize_root, human,
                    #                                'visibility',
                    #                                'Camera_B' + str(idx + 1),
                    #                                '{}.npy'.format(vizmap_idx))
                    # input_vizmap = np.load(vizmap_path).astype(np.bool)
                    # tmp_vizmaps.append(torch.from_numpy(input_vizmap))

                    input_vizmap = np.ones((6890)).astype(np.bool)
                    tmp_vizmaps.append(torch.from_numpy(input_vizmap))

                # load depth_map (already 512x512)
                if cfg.depth_map:
                    depthmap_pth = os.path.join(cfg.depth_root, human, 'Posing', cam_name,
                                                    filename.replace('.jpg', '.pt'))
                    depth_map = torch.load(depthmap_pth)
                    tmp_depthmap.append(depth_map.permute(1,2,0))

                # 
                in_K = np.array(self.cams[human]['K'][idx]).astype(np.float32)
                in_D = np.array(self.cams[human]['D'][idx]).astype(np.float32)

                # undistort 
                input_img = cv2.undistort(input_img, in_K, in_D)
                input_msk = cv2.undistort(input_msk, in_K, in_D)

                in_R = np.array(self.cams[human]['R'][idx]).astype(np.float32)
                in_T = (np.array(self.cams[human]['T'][idx]) / 1000.).astype(
                    np.float32)

                # resize
                input_img = cv2.resize(input_img, (W, H),
                                       interpolation=cv2.INTER_AREA)
                input_msk = cv2.resize(input_msk, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                
                if self.split == 'train' and 'RANDBG' in cfg.exp_name:
                    pass
                else:
                    if cfg.mask_bkgd:
                        if cfg.white_bkgd:
                            input_img[input_msk == 0] = 1
                        else:
                            input_img[input_msk == 0] = 0
            
                input_msk = (
                            input_msk != 0)  # bool mask : foreground (True) background (False)

                if cfg.use_viz_test and cfg.use_fg_masking:
                    if cfg.ratio == 0.5:
                        border = 5
                    kernel = np.ones((border, border), np.uint8)
                    input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                          kernel)

                # [-1,1]
                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()

                in_K[:2] = in_K[:2] * cfg.ratio

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
        
        # print(input_imgs[0].mean(), input_imgs[0].sum())
        # print(input_msks[0].mean(), input_msks[0].sum())
        # these are temporal aware 
        # print(len(input_imgs)) # 2
        # print(len(input_msks)) # 2
        # print(len(input_vizmaps)) # 2
        # print(len(input_depthmaps)) # 2
        # print(len(input_img_pths)) # 2
        # print(len(input_K)) # 2
        # print(len(input_R)) # 2
        # print(len(input_T)) # 2
        # these are not temporal aware. 
        # input_K = torch.stack(input_K)
        # input_R = torch.stack(input_R)
        # input_T = torch.stack(input_T)
        # print(input_K.shape, input_R.shape, input_T.shape)
        # # torch.Size([3, 3, 3]) torch.Size([3, 3, 3]) torch.Size([3, 3, 1])
      
        i = int(frame[:-4])
        feature, coord, out_sh, can_bounds, bounds, \
        Rh, Th, center, rot, trans, tmp_smpl_vertices, smpl_vertice_smplcoord, \
        min_xyz_smpl, max_xyz_smpl, blended_mtx, poses, shapes = self.prepare_input(human, i)
        
        # SMPL for current frame, (SMPL is shared across multi-view images)
        if cfg.time_steps == 1:
            smpl_vertices.append(tmp_smpl_vertices)

        # print(len(smpl_vertices)) # 2
        # print(tmp_smpl_vertices.shape) # (6890, 3)
        # print(self.nrays, self.split) 1024, train
        # print(bounds) # 3d bounding box under SMPL coord. [[x_min,y_min,z_min],[x_max,y_max,z_max]]
        # print(K.shape, R.shape, T.shape) # (3, 3) (3, 3) (3, 1)
        # K,R,T is the parameters for target view. 
        
        if self.split == 'train' and  'PATCHSAMPLE' in cfg.exp_name:
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
            
        # print(rgb.shape. ray_o.shape, near.shape, coord_.shape, mask_at_box.shape)
        # print(rgb.shape)   (1024, 3)
        # print(near) # near and far is different for each ray .
        # print(acc.shape) # 1024, 
        # print(acc) # [ 1 1 1 ... 0 1 0]
        # print(mask_at_box.shape) # 1024 

        ret = {
            'smpl_vertice': smpl_vertices, # smpl vertices for all time_step under world coordinate. (used in paint_neural_human )
            'smpl_vertice_smplcoord': smpl_vertice_smplcoord, # smpl under smplcoord for current target time step (used for PE_obs)
            'feature': feature, # 
            'coord': coord, # voxelized coordinate for SMPL vertex.  under SMPL coordinate. 
            'out_sh': out_sh, # 
            'min_xyz_smpl': min_xyz_smpl,
            'max_xyz_smpl':max_xyz_smpl
        }

        ret.update(sampled)
        # rvec to rotation matrix. 
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # print(img_path)
        # print(i) # 102
        # print(frame_index) 
    
        # i = int(os.path.basename(img_path)[:-4])
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]
            
        meta = {
            'bounds': bounds, # 3d bounding box under SMPL coord.
            'R': R, # R for target smpl 
            'Th': Th, # Th for target smpl 
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'cam_ind': cam_ind,
            'frame_index': frame_index,
            'human_idx': human_idx,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_depthmaps': input_depthmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'target_K': target_K,
            'target_R': target_R,
            'target_T': target_T,
            'blended_mtx': blended_mtx, # the blended rotation matrix from canonical to observation SMPL 
            'poses': poses, # poses for target smpl 
            'shapes': shapes, # shapes for target smpl 
            'img_path': img_path,
            'input_img_paths': input_img_pths
        }
        ret.update(meta)

        if self.split == 'train' and 'RANDBG'in cfg.exp_name:
            ret.update({'bgcolor': bgcolor / 255.}) # the nerf output is 0-1
            
        return ret

    def __len__(self):
        return len(self.ims)
