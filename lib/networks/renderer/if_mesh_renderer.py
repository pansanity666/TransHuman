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
import mcubes

from lib.networks.renderer.if_clight_renderer import Renderer as Base_Renderer


class Renderer(Base_Renderer):
    
    def __init__(self, net):
        super(Renderer, self).__init__(net)   
    
    def render(self, batch):
        
        pts = batch['pts']
        self.device = pts.device

        ### get valid pts by distance to SMPL 
        sh = pts.shape # torch.Size([1, 184, 168, 358, 3])
        dist, idx, Vnn = knn_points(pts.flatten(1,3), batch['tar_smpl_vertice'], K=1, return_nn=True)
        dist = dist.sqrt()
        pts_mask = (dist < 0.1).squeeze(-1) # torch.Size([1, 11066496])
        pts_mask = pts_mask.view(*sh[:-1])

        xyz = pts.clone().flatten(1,3) # query pts under world coordinate
        pts = self.world2smpl(pts, batch['Rh'], batch['Th']) 
        
        ### dummy videdir, the inferred RGB is actually not used. 
        viewdir = torch.zeros((*xyz.shape[:2], 27)).to(self.device)
        
        ### Paint SMPL 
        temporal_holders, temporal_weights, temporal_pixel_feat_map, temporal_pixel_feat_scale = self.get_painted_SMPL(batch)
        
        ### Process with TransHE
        all_holders = self.process_holders(temporal_holders)
        grouped_holders = self.can_body_grouping(all_holders)
        grouped_PE_can = self.voxel_PE_can.to(self.device).unsqueeze(0).repeat(grouped_holders.shape[0],1,1) # [T*V, 2115, 3]
        holder_completed = self.net.ViT(grouped_holders, self.normalize_PE(grouped_PE_can), mask=None,)
      
        ### Prepare DPaRF inputs: 
        DPaRF_param_dict = {'pts_smplcoord': pts.flatten(1,3),
                          'can_smpl_smplcoord': grouped_PE_can, # Currently not used. 
                          'obs_smpl_smplcoord': self.voxelization(self.dict_voxel2pc_ind, batch['tar_smpl_vertice_smplcoord'][0]).unsqueeze(0).repeat(sh[0], 1, 1), # bs, N_c, 3 
                          'blend_mtx': self.voxelization(self.dict_voxel2pc_ind, batch['blend_mtx'][0]).unsqueeze(0).repeat(sh[0], 1, 1, 1), # Bs, N_c, 4, 4
                          'input_blend_mtx': [self.voxelization(self.dict_voxel2pc_ind, bm[0]).unsqueeze(0).repeat(sh[0], 1, 1, 1) for bm in batch['input_blend_mtx']],
                          'input_smpl_smplcoord': [self.voxelization(self.dict_voxel2pc_ind, s[0]).unsqueeze(0).repeat(sh[0], 1, 1) for s in batch['input_smpl_vertice_smplcoord']]
                          }
      
        ### NeRF MLP forward. 
        face_idx = torch.tensor(self.faces.astype(int)).long()
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
                                pts_mask=pts_mask.flatten(1,3) if pts_mask is not None else None 
                                ) 

        ### marching cube 
        alpha = raw[...,-1:]
        cube = alpha[0, :, 0].view(*sh[1:4]).detach().cpu().numpy()
        cube = np.pad(cube, 10, mode='constant')
        data_name = cfg.data_root.split('/')[-1]
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)

        ### vertices from index coordinate to world coordinate. 
        can_bounds = batch['can_bounds'][0].cpu().numpy() # 2, 3 
        LB = (can_bounds[0] - 10*np.array(cfg.voxel_size))[None,...]
        vertices_world = vertices * np.array(cfg.voxel_size)[None,...] + LB
        mesh = trimesh.Trimesh(vertices_world, triangles)
        
        ret = {'cube': cube, 'mesh': mesh}
        
        return ret

    
    
    
    