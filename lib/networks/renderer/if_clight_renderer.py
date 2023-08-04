import torch
import torch.nn as nn
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import matplotlib.pyplot as plt
import numpy as np
import gc
import math
import time

from lib.utils.vis_utils import plotPoints, write_point_cloud
from spconv.pytorch.utils import PointToVoxel
import torch.distributed as dist

from lib.utils.base_utils import euclidean_dist, batch_rod2quat, gaussion, solve_lsr
from lib.utils.render_utils import barycentric_mapping, LBS_mapping

from collections import defaultdict
from pytorch3d.ops import knn_points
import pickle
import chumpy
import trimesh
import tempfile
import os
import open3d as o3d

import shutil
import random

import re 
from lib.utils.SMPL import SMPL
from lib.networks.vision_transformer import RandomMaskingGenerator

import torchvision

class Renderer:

    def __init__(self, net):
        self.net = net

        # load vertex canonical 
        canonical_SMPL = './data/smplx/smpl/SMPL_NEUTRAL.pkl'
        with open(canonical_SMPL, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        vertex_can = data['v_template']
        self.faces = data['f']
        self.vertex_can = torch.tensor(vertex_can).contiguous() # 6890, 3

        self.CR = torch.tensor([-1.5,-1.5,-1.5,1.5,1.5,1.5]) # bounding box used for PE normalization 

        ### KMeans dict 
        # num_voxel = int(re.search('KMEANS(\d+)', cfg.exp_name).group(1))
        num_voxel = cfg.num_class
        self.pc2voxel_ind, self.dict_voxel2pc_ind = np.load(f'./kmeans_dict/kmeans_dict_{num_voxel}.npy', allow_pickle=True).item().values()
        self.avg_pc_orders = torch.tensor([torch.mean(torch.tensor(pc_inds, dtype=torch.float32)) for pc_inds in self.dict_voxel2pc_ind.values()])
        self.pc2voxel_ind = torch.tensor(self.pc2voxel_ind).type(torch.int64)
    
        ### (optional) plot voxel hist 
        # all_l = []
        # for k,v in self.dict_voxel2pc_ind.items():
        #     all_l.append(len(v))
        #     # print(k,v)
        # plt.rc('font',family='Times New Roman')
        # plt.style.use('ggplot')
        # plt.hist(np.array(all_l), 10, density=True)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Vertex Number', fontsize=20)
        # plt.ylabel('Probability', fontsize=20)
        # plt.savefig('./hist_grid.jpg', bbox_inches='tight')
        
        self.voxel_PE_can = self.voxelization(self.dict_voxel2pc_ind, self.vertex_can) # 300, 3
        
    def get_relative_depth(self, depth, uv, batch, det=0.07, t=0):
        ### get relative depth toward each input view (depth - smpl_surface_depth)
        # depth: 3, 6890, 1, depth toward each input view. 
        # uv: 3, 6890, 2, uv toward each input view. 
        
        assert depth.shape[1] == uv.shape[1]
        
        input_depthmaps = batch['input_depthmaps'][t] # torch.Size([1, 3, 512, 512, 1]) 
        input_depthmaps = input_depthmaps.permute(0,1,4,2,3).flatten(0,1) # 3, 1, 512, 512  

        uv_normed = (uv / input_depthmaps.shape[2]) * 2. - 1. # [0,511] -> [-1,1]
        
        surface_depth = F.grid_sample(input_depthmaps, uv_normed.unsqueeze(1)) # torch.Size([3, 1, 1, 6890]) b, C, hout, wout
        surface_depth = surface_depth.permute(0,3,1,2).squeeze().unsqueeze(-1) # 3, 6890, 1

        vis_mask = (depth <= surface_depth + det)[..., 0] # 3, 6890
        relative_depth = depth - (surface_depth + det)

        return surface_depth, vis_mask, relative_depth

    def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale, prev_weight=None, prev_holder=None, vis=False):
        ### this function paints reference img features to SMPL vertex.

        ## SMPL vertices under world coord of target view 
        smpl_vertice = batch['input_smpl_vertice'][t] # 1, 6890, 3
        num_vertices = smpl_vertice.shape[1]
        
        ### Optionally augmenta SMPL vertice 
        if cfg.rasterize:
            # the mask for SMPL vertexes. 
            vizmap = batch['input_vizmaps'][t] # torch.Size([1, 3, 6890]) # print(vizmap.sum()) 7448 # bs, 3, 6890

        image_shape = batch['input_imgs'][t].shape[-2:] # 512, 512
 
        # R, T, K for input views
        input_R = batch['input_R'][t]
        input_T = batch['input_T'][t]
        input_K = batch['input_K'][t]

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)

        if cfg.rasterize:
            result = vizmap[0]

        ## project SMPL vertex to image plane coord (uv).
        # i.e., project smpl vertices to reference views for fetching pixel-aligned features. 
        vertice_rot = torch.matmul(input_R[:, None], smpl_vertice.unsqueeze(-1))[..., 0]
        vertice = vertice_rot + input_T[:, None, :3, 0] # (1, 6890, 3) + (3, 1, 3) 
        vertice = torch.matmul(input_K[:, None], vertice.unsqueeze(-1))[..., 0] # torch.Size([3, 1, 3, 3]) x torch.Size([3, 6890, 3])
        uv = vertice[:, :, :2] / vertice[:, :, 2:] # torch.Size([3, 6890, 2])

        ### retrive surface depth of each vertex
        if cfg.depth_map and cfg.depth_vizmap: 
            det = 0.07 # 0.07
            depth = vertice[:, :, 2:] 
            surface_depth, vis_mask, relative_depth = self.get_relative_depth(depth, uv, batch, det=det, t=t)
            result = vis_mask
             
        ### Visualize painted SMPL
        if vis:
            # visualize input images 
            torchvision.utils.save_image(batch['input_imgs'][t][0][0:1], './input_img_T{}.jpg'.format(t))
            
            # visualize uv map
            project_img = torch.zeros((512,512))
            for h,w in uv[0]:
                try:
                    project_img[int(h), int(w)] = 1 
                except: pass 
            # torchvision.utils.save_image(project_img.unsqueeze(0).unsqueeze(0), './uv_test.jpg')
            torchvision.utils.save_image(project_img.unsqueeze(0).unsqueeze(0).cpu() * 0.5 + \
                                        batch['input_imgs'][t][0][0:1].cpu() *0.5, \
                                        './input_img_with_uv_T{}.jpg'.format(t))

            # visualize point clouds 
            pixel_color = self.sample_from_feature_map(batch['input_imgs'][t][0],
                                                    holder_feat_scale, 
                                                    image_shape,
                                                    uv)
            pixel_RGB = pixel_color[0].t() # 6890, 3 
            pixel_BGR = pixel_RGB[:,[2,1,0]] # 6890, 3 
            pixel_BGR = pixel_BGR * 255 # 0,1 -> 0,255
            _dir = "./painted_SMPL_T{}.ply".format(t)
            os.makedirs(os.path.dirname(_dir), exist_ok=True)
            # write_point_cloud(_dir, torch.cat([smpl_vertice[0], pixel_BGR], dim=1).cpu().numpy(), )

            vis = torch.cat([smpl_vertice[0][result[0]], pixel_BGR[result[0]]], dim=1)
            invis = torch.cat([smpl_vertice[0][~result[0]], torch.zeros_like(pixel_BGR[~result[0]])], dim=1)
            write_point_cloud(_dir, torch.cat([vis, invis], dim=0).cpu().numpy())
            
    
        latent = self.sample_from_feature_map(holder_feat_map,
                                              holder_feat_scale, 
                                              image_shape,
                                              uv)
        latent = latent.permute(0, 2, 1) # torch.Size([3, 6890, 64])
        num_input = latent.shape[0]
        

        if cfg.rasterize:
            final_result = result
        else:
            nview, nvertex = latent.shape[:2]
            final_result = torch.ones((nview, nvertex)).cuda() # print(final_result.shape, final_result.sum()) # torch.Size([3, 6890]) tensor(7377, device='cuda:0')
        big_holder = torch.zeros((latent.shape[0], latent.shape[1], cfg.embed_size)).cuda() # [3, 6890, 64]
        big_holder[final_result == True, :] = latent[final_result == True, :] # print(final_result.shape, big_holder.shape) # torch.Size([3, 6890]) torch.Size([3, 6890, 64])
        
        return final_result, big_holder

    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):
        
        # feat_map: (3, 192, 256, 256)
        # feat_scale: image_size / feat_map_size, e.g.,  (2,)
        # image_shape: (512, 512)
        # uv: projected coords ranged in the original image size. # (3, 6890, 2 )
        
        scale = feat_scale / image_shape # 2 / 512 
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())
     
        uv = uv * scale - 1.0 # [0, 511] -> [-1, 1] # ( x * 2 / H -1; y * 2 / W - 1)
        uv = uv.unsqueeze(2) # torch.Size([3, 6890, 1, 2])
        
        samples = F.grid_sample(
            feat_map,
            uv,    # the coordinate should be normalized to [-1, 1]
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map, pixel_feat_scale, batchify=False, t=None, vis=False):
        # Only get target time_step PN feats ATM. 
        # xyz: [B, N, 3] or [B, nray, npts, 3]
        
        image_shape = batch['input_imgs'][t].shape[-2:]
        input_R = batch['input_R'][t]
        input_T = batch['input_T'][t]
        input_K = batch['input_K'][t]

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)
   
        # project xyz to image plane. 
        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3)
        xyz_input = xyz
        
        xyz = repeat_interleave(xyz, input_R.shape[0])
        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        uv = xyz[:, :, :2] / xyz[:, :, 2:] # torch.Size([3, 32768, 2])
  
        # ############  Visualize painted xyz + SMPL ############
        if vis:
            # torchvision.utils.save_image(batch['input_imgs'][t][0][0:1], './input_img_T{}.jpg'.format(t))
            project_img = torch.zeros((512,512))
            for h,w in uv[0]:
                try:
                    project_img[int(h), int(w)] = 1 
                except: pass
            # torchvision.utils.save_image(project_img.unsqueeze(0).unsqueeze(0), './uv_test.jpg')
            torchvision.utils.save_image(project_img.unsqueeze(0).unsqueeze(0).cpu() * 0.5 + \
                                        batch['input_imgs'][t][0][0:1].cpu() *0.5, \
                                        './input_img_with_uv_T{}.jpg'.format(t))
            
            
            pixel_color = self.sample_from_feature_map(batch['input_imgs'][t][0],
                                                    pixel_feat_scale, 
                                                    image_shape,
                                                    uv)
            pixel_RGB = pixel_color[0].t() # 6890, 3 
            pixel_BGR = pixel_RGB[:,[2,1,0]] # 6890, 3 
            pixel_BGR = pixel_BGR * 255 # 0,1 -> 0,255
            _dir = "./painted_xyzSMPL_T{}.ply".format(t)
            os.makedirs(os.path.dirname(_dir), exist_ok=True)
            xyz_color = torch.cat([xyz_input[0], pixel_BGR], dim=1) 
            # write_point_cloud(_dir, xyz_color.cpu().numpy(), )
            smpl_color = torch.cat([batch['input_smpl_vertice'][t][0], torch.zeros_like(batch['input_smpl_vertice'][t][0]) + torch.tensor([0,0,255]).unsqueeze(0).cuda()], dim=1)
            # write_point_cloud(_dir, smpl_color.cpu().numpy(), )
            write_point_cloud(_dir, torch.cat([xyz_color, smpl_color], dim=0).cpu().numpy(),)

        # 3, 256, 65536
        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, 
                                                  image_shape,
                                                  uv)

        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def world2smpl(self, pts, Rh, Th):
        """transform pts from the world coordinate to the smpl coordinate"""
        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[-1]) # Bs, N_pts, 3 
        pts = pts - Th
        pts = torch.matmul(pts, Rh)
        return pts.view(*sh)
    
    def smpl2world(self, pts, Rh, Th):
        # inverse of world2smpl 
        # y = (x - T) * R --> x = y*R^-1 + T
        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[-1]) # Bs, N_pts, 3 
        inv_Rh = torch.inverse(Rh)
        pts = torch.matmul(pts,inv_Rh) + Th
        return pts.view(*sh)

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts


    def get_grid_coords(self, pts, sp_input, batch):
        # get grid coords for sampled points. 
        # pts is under SMPL coord

        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]] # xyz -> zyx
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        # print(dhw.shape, min_dhw.shape) # torch.Size([1, 1024, 64, 3]) torch.Size([1, 3])

        dhw = dhw - min_dhw[:, None, None]
        # print(cfg.voxel_size) # [0.005, 0.005, 0.005]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1 # [0, 1] -> [-1, 1]

        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]

        return grid_coords

    def voxel_pc_order(self, pc2voxel_ind):
        
        # pc2voxel_ind: (num_points,)
        # return: (num_voxels,) 
        dict_voxel2pc_ind = defaultdict(list)
        for pc_ind, voxel_ind in enumerate(pc2voxel_ind):
            dict_voxel2pc_ind[voxel_ind.cpu().item()].append(pc_ind)
        # sort dict_voxel2pc_ind by voxel_ind
        dict_voxel2pc_ind = dict(sorted(dict_voxel2pc_ind.items()))
        avg_pc_orders = torch.tensor([torch.mean(torch.tensor(pc_inds, dtype=torch.float32)) for pc_inds in dict_voxel2pc_ind.values()])

        return avg_pc_orders, dict_voxel2pc_ind
        
    def voxelization(self, dict_voxel2pc_ind, SMPL, return_list=False):
        # voxelize based on dict
    
        voxel_list = []
        voxel = []

        for vn, pc_list in dict_voxel2pc_ind.items():
            # mean pooling 
            voxel.append(SMPL[pc_list].mean(0))
            voxel_list.append(SMPL[pc_list])
        voxel = torch.stack(voxel) # torch.Size([2115, 67])
   
        if return_list:
            return voxel, voxel_list
        else:
            return voxel

    def normalize_PE(self, PE, CR=None):
        assert len(PE.shape) == 3
        # Normalize Position via CR
        if CR is None:
            min_xyz, max_xyz = self.CR[:3][None, None, :].to(PE.device), self.CR[3:][None, None, :].to(PE.device)
        else:
            min_xyz, max_xyz = CR[:3][None, None, :].to(PE.device), CR[3:][None, None, :].to(PE.device)

        PE_normed = (((PE - min_xyz) / (max_xyz - min_xyz)) - 0.5) * 2
        PE_normed = PE_normed.type(torch.float32).to(PE.device)
        return PE_normed
        
    def get_painted_SMPL(self, batch):
        # return: list of length T 
        image_list = batch['input_imgs']
        weight = None
        holder = None
        temporal_holders = []
        temporal_weights = []
        temporal_pixel_feat_map = []
        temporal_pixel_feat_scale = []
        for t in range(cfg.time_steps): # time_steps is always 1 in our experiments. 
            
            ### embed reference images  
            images = image_list[t].reshape(-1, *image_list[t].shape[2:]) # torch.Size([3, 3, 512, 512])
            # torchvision.utils.save_image(images.cpu(), './test_{}.jpg'.format(t))
            holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(images)
            temporal_pixel_feat_map.append(pixel_feat_map)
            temporal_pixel_feat_scale.append(pixel_feat_scale)

            ### paint the holder
            weight, holder = self.paint_neural_human(batch, t, holder_feat_map, holder_feat_scale, weight, holder, vis=False)
            
            temporal_holders.append(holder)
            temporal_weights.append(weight)
        
        return temporal_holders, temporal_weights, temporal_pixel_feat_map, temporal_pixel_feat_scale
    
    def process_holders(self, temporal_holders):
        assert cfg.time_steps == 1 # Only support T=1 ATM. 
        return temporal_holders[0]

    def can_body_grouping(self, all_holders):
        # all_holders: T*V, 6890, D 
        # return: T*V, N_c, D 
 
        grouped_holders = []
        voxels_PE_can_list = []
        for _holder in all_holders:
            # _holder: 6890, 192    
            grouped_SMPL = self.voxelization(self.dict_voxel2pc_ind, _holder)
            grouped_holders.append(grouped_SMPL)
        grouped_holders = torch.stack(grouped_holders) # [3, 2115, 768]
        
        return grouped_holders
            
    def render_fast(self, batch, is_train=True):
        ### fast rendering, only called during inference stage. 
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
    
        ### mask out valid rays and close points 
        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        sh = pts.shape # 1, 88059, 64, 3
      
        dist, idx, Vnn = knn_points(pts.flatten(1,2), batch['tar_smpl_vertice'], K=1, return_nn=True)
        dist = dist.sqrt()
        valid_pts_mask = (dist < 0.1).squeeze(-1) # 1, 5635776 
        valid_pix_mask = valid_pts_mask.view(1, *sh[1:3]).sum(-1) > 0 # 1, 88059
        valid_pix_pts_mask = valid_pts_mask.view(1, *sh[1:3])[valid_pix_mask].unsqueeze(0)  # torch.Size([1, 69416, 64])
  
        ### (optional) visualization of bbox-based redering (original) vs. smpl-based (ours) 
        # mask_at_box = batch['mask_at_box'][0] 
        # pred_img = torch.zeros((512, 512, 1)).view(-1, 1)
        # valid_pix = torch.zeros((valid_pix_mask.shape[1], 1))
        # valid_pix[valid_pix_mask[0]] = 1 
        # pred_img[mask_at_box] = valid_pix
        # import torchvision
        # torchvision.utils.save_image(pred_img.view(512,512,1).permute(2,0,1).unsqueeze(0), './valid_mask_closepts.jpg')
        # pred_img = torch.zeros((512, 512, 1)).view(-1, 1)
        # pred_img[mask_at_box] = 1 
        # torchvision.utils.save_image(pred_img.view(512,512,1).permute(2,0,1).unsqueeze(0), './valid_mask_bbox.jpg')
        
        ### nference close points 
        batch['ray_o'] = batch['ray_o'][valid_pix_mask].unsqueeze(0)
        batch['ray_d'] = batch['ray_d'][valid_pix_mask].unsqueeze(0)
        batch['near'] = batch['near'][valid_pix_mask].unsqueeze(0)
        batch['far'] = batch['far'][valid_pix_mask].unsqueeze(0)

        pts = pts[valid_pix_mask].unsqueeze(0) # torch.Size([1, 69416, 64, 3])
        z_vals = z_vals[valid_pix_mask].unsqueeze(0) # torch.Size([1, 69416, 64])
        
        ### init return
        ret_depth_map = torch.zeros((1, sh[1])).to(pts.device)
        ret_rgb_map = torch.zeros((1, sh[1], 3)).to(pts.device)
        ret_acc_map = torch.zeros((1, sh[1])).to(pts.device)
        
        ret = self._render(batch, pts, z_vals, is_train=is_train, pts_mask=valid_pix_pts_mask)
 
        ret_depth_map[valid_pix_mask] = ret['depth_map'][0]
        ret_rgb_map[valid_pix_mask] = ret['rgb_map'][0]
        ret_acc_map[valid_pix_mask] = ret['acc_map'][0]
        
        ret = {
            'depth_map': ret_depth_map,
            'rgb_map': ret_rgb_map,
            'acc_map': ret_acc_map
        }

        return ret 
    
    def render(self, batch, is_train=True): 
        ### Called during training stage. 
        ray_o = batch['ray_o'] # torch.Size([1, 99456, 3])
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far) # print(z_vals.shape) # torch.Size([1, 1024, 96])
        
        ### original 
        ret = self._render(batch, pts, z_vals, is_train=is_train)

        return ret 

    def _render(self, batch, pts, z_vals, is_train=True, pts_mask=None):

        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']  
        sh = ray_o.shape

        self.device = ray_o.device
        self.is_train = is_train 
  
        ### prepare pts and viewdirs
        bs, nray, npix = pts.shape[:3] # 1, 1024, 64
        xyz = pts.clone().flatten(1,2)  # under world coordinate. 
        sincos_pts_worldcoord = embedder.xyz_embedder(pts) # sin-cos embedding # torch.Size([1, 1024, 64, 63])
        sincos_pts_worldcoord = sincos_pts_worldcoord.view(sh[0], -1, embedder.xyz_dim) # torch.Size([1, 65536, 63])
        
        ### (optional) plot sampled points and smpl under the world coordinate. 
        # write_point_cloud('./world_smpl_pts.ply', torch.cat([pts[0][::1000].flatten(0,1), batch['smpl_vertice'][0][0]], dim=0).cpu().numpy())
        
        pts = self.world2smpl(pts, batch['Rh'], batch['Th']) # from world coord to can coord # torch.Size([1, 1024, 64, 3])
        
        ### (optional) plot sampled points and smpl under the smpl coordinate.
        # write_point_cloud('./smpl_smpl_pts.ply', torch.cat([pts[0][::1000].flatten(0,1), batch['tar_smpl_vertice_smplcoord'][0]], dim=0).cpu().numpy())

        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()
        viewdir = viewdir.view(sh[0], -1, embedder.view_dim) # torch.Size([1, 65536, 27])

        ### Paint SMPL 
        temporal_holders, temporal_weights, temporal_pixel_feat_map, temporal_pixel_feat_scale = self.get_painted_SMPL(batch)
        
        ### Process with TransHE
        all_holders = self.process_holders(temporal_holders)
        grouped_holders = self.can_body_grouping(all_holders)
        grouped_PE_can = self.voxel_PE_can.to(self.device).unsqueeze(0).repeat(grouped_holders.shape[0],1,1) # [T*V, 2115, 3]
     
        holder_completed = self.net.ViT(grouped_holders, self.normalize_PE(grouped_PE_can), mask=None,)
      
        ### Prepare DPaRF inputs: 
        DPaRF_param_dict = {'pts_smplcoord': pts.flatten(1,2),
                          'can_smpl_smplcoord': grouped_PE_can, # Currently not used. 
                          'obs_smpl_smplcoord': self.voxelization(self.dict_voxel2pc_ind, batch['tar_smpl_vertice_smplcoord'][0]).unsqueeze(0).repeat(sh[0], 1, 1), # bs, N_c, 3 
                          'blend_mtx': self.voxelization(self.dict_voxel2pc_ind, batch['blend_mtx'][0]).unsqueeze(0).repeat(sh[0], 1, 1, 1), # Bs, N_c, 4, 4
                          'input_blend_mtx': [self.voxelization(self.dict_voxel2pc_ind, bm[0]).unsqueeze(0).repeat(sh[0], 1, 1, 1) for bm in batch['input_blend_mtx']],
                          'input_smpl_smplcoord': [self.voxelization(self.dict_voxel2pc_ind, s[0]).unsqueeze(0).repeat(sh[0], 1, 1) for s in batch['input_smpl_vertice_smplcoord']]
                          }

        ### NeRF MLP forward. 
        face_idx = torch.tensor(self.faces.astype(int)).long()
        if ray_o.size(1) <= 2400: 
            ### used during train 
            temporal_pixel_feat, temporal_visibility = [], []
            temporal_xyz_smplcoord = []
            
            assert cfg.time_steps == 1 
            T = cfg.time_steps
            for t in range(T):
                _xyz = xyz
                temporal_xyz_smplcoord.append(pts.flatten(1,2))
                pixel_feat = self.get_pixel_aligned_feature(batch, _xyz, temporal_pixel_feat_map[t], temporal_pixel_feat_scale[t], t=t)
                temporal_pixel_feat.append(pixel_feat)

            pixel_feat = torch.stack(temporal_pixel_feat).flatten(0,1) # T*V, 256, 65536

            raw = self.net(pixel_feat=pixel_feat, 
                           sincos_viewdir=viewdir,
                           DPaRF_param_dict=DPaRF_param_dict,
                           holder=holder_completed,
                           face_idx=face_idx, # BM 
                           )
        else:
            ### used during test. 
            raw = self.batchify_rays(
                                     chunk=1024 * 32,  # 1024 * 32 
                                     net_c=None, 
                                     batch=batch, 
                                     xyz=xyz, 
                                     temporal_pixel_feat_map=temporal_pixel_feat_map,
                                     temporal_pixel_feat_scale=temporal_pixel_feat_scale,
                                     holder=holder_completed,
                                     sincos_viewdir=viewdir,
                                     DPaRF_param_dict=DPaRF_param_dict,
                                     face_idx=face_idx,
                                     pts_mask=pts_mask.flatten(1,2) if pts_mask is not None else None 
                                     ) 
            
        ### accumulating 
        raw = raw.reshape(-1, z_vals.size(2), 4)
        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        rgb_map = rgb_map.view(*sh[:-1], -1)
        acc_map = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])

        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()

        return ret
    
    def batchify_rays(self,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      temporal_pixel_feat_map=None,
                      temporal_pixel_feat_scale=None,
                      sincos_viewdir=None,
                      DPaRF_param_dict=None,
                      holder=None,
                      face_idx=None,
                      pts_mask=None
                      ):
        """Render rays in smaller minibatches to avoid OOM.
        """
        pts_smplcoord = DPaRF_param_dict['pts_smplcoord']
        
        # loop of  chunk 
        all_ret = []
        for i in range(0, pts_smplcoord.shape[1], chunk):
            
            bs = xyz.shape[0]
            temporal_pixel_feat, temporal_visibility = [], []
            temporal_xyz_smplcoord = []
            
            assert cfg.time_steps == 1 
            T = cfg.time_steps 
            for t in range(T):
                _xyz = xyz[:, i:i+chunk]
                temporal_xyz_smplcoord.append(pts_smplcoord[:, i:i+chunk] if pts_smplcoord is not None else None)
                pixel_feat = self.get_pixel_aligned_feature(batch, _xyz, temporal_pixel_feat_map[t], temporal_pixel_feat_scale[t], t=t)
                temporal_pixel_feat.append(pixel_feat)
   
            pixel_feat = torch.stack(temporal_pixel_feat).flatten(0,1) # T*V, 256, chunk
            DPaRF_param_dict['input_pts_smplcoord'] = temporal_xyz_smplcoord
            
            ### Chunk dict 
            DPaRF_param_dict['pts_smplcoord'] = pts_smplcoord[:, i:i + chunk] if pts_smplcoord is not None else None
            ret = self.net(pixel_feat=pixel_feat, 
                           sincos_viewdir=sincos_viewdir[:, i:i + chunk] if sincos_viewdir is not None else None,
                           DPaRF_param_dict=DPaRF_param_dict,
                           holder=holder,
                           face_idx=face_idx,
                           pts_mask=pts_mask[:, i:i + chunk] if pts_mask is not None else None
                           )
            all_ret.append(ret)
            
        all_ret = torch.cat(all_ret, 1)

        return all_ret
