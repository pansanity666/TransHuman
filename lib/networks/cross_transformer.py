import torch.nn as nn
import torch.nn.functional as F
import torch

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

from lib.config import cfg
from lib.networks.encoder import SpatialEncoder
import math
import time

import lib.networks.vision_transformer as ViT
from lib.networks.vision_transformer import trunc_normal_, PositionalEncoding

from lib.utils.base_utils import euclidean_dist, get_quaternion, count_parameters
import re 

import numpy as np

from pytorch3d.ops import knn_points

class SpatialKeyValue(nn.Module):

    def __init__(self, input_dim=256, att_dim=128, out_dim=256):
        super(SpatialKeyValue, self).__init__()
        self.key_embed = nn.Conv1d(input_dim, att_dim, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(input_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        return (self.key_embed(x),
                self.value_embed(x))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def combine_interleaved(t, num_input=4, agg_type="average"):

    t = t.reshape(-1, num_input, *t.shape[1:])

    if agg_type == "average":
        t = torch.mean(t, dim=1)

    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # ViT
        self.ViT = ViT.vit_tiny(depth=cfg.vit_depth)
        
        # update embed_size based on ViT type
        cfg.embed_size = self.ViT.embed_dim

        # Encoder
        self.encoder = SpatialEncoder()
        
        # Cross-attention 
        self.spatial_key_value_0 = SpatialKeyValue()
        self.spatial_key_value_1 = SpatialKeyValue()

        # SparseConvNet (please ignore this one, not used)
        self.xyzc_net = SparseConvNet() # 18.15MB

        # ReLU 
        self.actvn = nn.ReLU()
    
        self.PE_relative = PositionalEncoding(num_freqs=cfg.KNN_FREQ) # PE for local coordinates 
        self.fc_0 = nn.Conv1d(cfg.embed_size + self.PE_relative.d_out, 256, 1) # mapping from human representation to cross-attention input 
    
        ### NeRF MLPs
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)

        self.view_fc = nn.Conv1d(283, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.fc_3 = nn.Conv1d(256, 256, 1)
        self.fc_4 = nn.Conv1d(128, 128, 1)

        ## NeRF MLPs
        cfg.img_feat_size = 256 + 128 
        self.alpha_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.rgb_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.rgb_res_1 = nn.Conv1d(cfg.img_feat_size, 128, 1)
            
    def cross_attention(self, holder, pixel_feat,):
        # holder: 3, 256, 65536; (1, 256, 65536) is also supported. 
        # pixel_feat: 3, 256, 65536
        # calculate attention from difference views. 
        # return: 3, 256, 65536

        key_embed, value_embed = self.spatial_key_value_0(
            pixel_feat.permute(2, 1, 0))

        query_key, query_value = self.spatial_key_value_1(
            holder.permute(2, 1, 0))
      
        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2), query_key)
        A = A / math.sqrt(k_emb) # 65536, 3, 3 
        
        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)

        final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0) 

        return final_holder

    def get_dist_weight(self, knn_dist):
        
        knn_dist = - knn_dist # 65536, 10
        knn_dist_normed = F.softmax(knn_dist/cfg.KNN_DIST_ALPHA, dim=1)
        
        return knn_dist_normed
    
    def get_human_representation(self, pts_smplcoord, obs_smpl_smplcoord, blend_mtx, holder):
        ### Get human representation using DPaRF. 
        # pts_smplcoord: (Bs, chunk[mask], 3)
        # obs_smpl_smplcoord: (Bs, N_c, 3)
        # blend_mtx: (Bs, N_c, 4, 4)
        # holder: (N_v, N_c, 192)
        # return: (N_v, 255, chunk[mask])
        
        pts_smplcoord = pts_smplcoord.view(-1, 3) # 65536, 3  
        obs_smpl_smplcoord = obs_smpl_smplcoord.view(-1, 3) # 900, 3  

        ### calculate KNN 
        knn_dist, knn_idx, _ = knn_points(pts_smplcoord.unsqueeze(0), obs_smpl_smplcoord.unsqueeze(0), K=cfg.KNN)
        knn_dist = knn_dist[0].sqrt()
        knn_idx = knn_idx[0]
        knn_dist_normed = self.get_dist_weight(knn_dist) # KNN aggregation weight. 

        ### calculate mask  
        mask_preserve = None
        if cfg.use_truncation:
            knn_dist_min = knn_dist.min(-1)[0] # 65536
            sigma = cfg.KNN_SIGMA
            mask_preserve = knn_dist_min < sigma # 65536

        ### get local coordinates under DPaRF
        closest_voxels_obs = obs_smpl_smplcoord[knn_idx] # 65536, k, 3
        relative_positions = pts_smplcoord.unsqueeze(1) - closest_voxels_obs # 65536, k, 3
        _rot_mtx = blend_mtx[0][...,:3,:3].type(torch.float32) # 900, 3, 3 # [r1, r2, r3]
        rot_mtx = _rot_mtx[knn_idx] # 65536, k, 3, 3
        deformed_pos = torch.matmul(relative_positions.unsqueeze(-2), rot_mtx) # 65536, k, 1, 3
        deformed_pos = deformed_pos.squeeze(-2) # 65536, k, 3 

        ### sin-cos embedding. 
        n, k = deformed_pos.shape[:2]
        deformed_pos = self.PE_relative(deformed_pos.view(-1,3)).view(n,k,-1) # 

        ### concat condition features & KNN aggregation 
        human_rep = []
        # for view in range(self.n_input_holder):
        for _holder in holder:
            _feat = _holder[knn_idx]  # 65536, 10, 192 
            _feat = torch.cat([_feat, deformed_pos], dim=-1) 
            _feat = torch.sum(knn_dist_normed.unsqueeze(-1) * _feat, dim=1)
            _feat = _feat.t().unsqueeze(0)
            human_rep.append(_feat)
        human_rep = torch.cat(human_rep, dim=0) # 3, 255, 65536
        
        return human_rep, mask_preserve
    
    def forward(self, 
                pixel_feat, 
                sincos_viewdir,
                DPaRF_param_dict,
                holder=None,  # (T*V, N_c, 192)
                face_idx=None, 
                pts_mask=None):
   
        pts_smplcoord = DPaRF_param_dict['pts_smplcoord'] # (Bs, Nc, 3)
        obs_smpl_smplcoord = DPaRF_param_dict['obs_smpl_smplcoord']
        blend_mtx = DPaRF_param_dict['blend_mtx'] # (Bs, Nc, 4, 4)
        # input_blend_mtx = DPaRF_param_dict['input_blend_mtx'] # T; (Bs, N_c, 4, 4) 
        # input_smpl_smplcoord = DPaRF_param_dict['input_smpl_smplcoord'] # T; (Bs, N_c, 3)
        # input_pts_smplcoord = DPaRF_param_dict['input_pts_smplcoord'] # T; (Bs, N_pts, 3)

        B = 1  # We assume Bs = 1 here. 
        self.n_input = int(pixel_feat.shape[0] / B)
        self.n_input_holder = int(holder.shape[0] / B)
        
        ##############################
        ### Mask pts based on distance 
        ##############################
        self.pts_mask = pts_mask
        if self.pts_mask is not None:     
            raw_temp = torch.zeros((B, pts_smplcoord.shape[1], 4)).to(pts_smplcoord.device)
            if self.pts_mask.sum() == 0:  
                return raw_temp
            pts_smplcoord = pts_smplcoord[self.pts_mask].unsqueeze(0)
            pixel_feat = pixel_feat[..., self.pts_mask[0]]
            sincos_viewdir = sincos_viewdir[:, self.pts_mask[0], :]
            # input_pts_smplcoord = [i[:,self.pts_mask[0],:] for i in input_pts_smplcoord]
  
        ####################################
        ### Get human representation 
        ####################################
        human_rep, mask_preserve = self.get_human_representation(pts_smplcoord, obs_smpl_smplcoord, blend_mtx, holder)
                
        ####################################
        ### NeRF MLP forward 
        ####################################
        def _forward(human_rep, pixel_feat, sincos_viewdir):
            # trunc points forward wrapper. 
            if cfg.use_truncation: # Not used by default 
                num_pts = pixel_feat.shape[-1]
                if mask_preserve.sum() > 0:
                    human_rep = human_rep[..., mask_preserve]
                    pixel_feat = pixel_feat[..., mask_preserve]
                    sincos_viewdir = sincos_viewdir[:, mask_preserve]
                    raw_preserved = self.MLP_forward(human_rep, pixel_feat, sincos_viewdir)
                    raw = torch.zeros((B, num_pts, 4)).to(human_rep.device)
                    raw[:,mask_preserve,:] = raw_preserved
                else:
                    # support batchify testing 
                    raw = torch.zeros((B, num_pts, 4)).to(human_rep.device)
            else:
                raw = self.MLP_forward(human_rep, pixel_feat, sincos_viewdir)
            
            return raw    

        raw = _forward(human_rep, pixel_feat, sincos_viewdir)
        if self.pts_mask is not None:
            raw_temp[self.pts_mask] = raw[0]
            raw = raw_temp 
                
        return raw

    def MLP_forward(self, human_rep, pixel_feat, sincos_viewdir):
        if self.pts_mask is not None:
            ### Fast rendering (inference)
            return self.MLP_forward_ori_progressive(human_rep, pixel_feat, sincos_viewdir)
        else:
            return self.MLP_forward_ori(human_rep, pixel_feat, sincos_viewdir)

    def MLP_forward_ori(self, human_rep, pixel_feat, sincos_viewdir):
        
        inter_net = self._multiview_agg(human_rep, pixel_feat)
        alpha = self._alpha_forward(inter_net)
        rgb = self._RGB_forward(inter_net, pixel_feat, sincos_viewdir)

        raw = torch.cat((rgb, alpha), dim=1) # 1, 4, 65536 
        raw = raw.transpose(1, 2) # 1, 65536, 4

        return raw
    
    def MLP_forward_ori_progressive(self, human_rep, pixel_feat, sincos_viewdir):
        
        inter_net = self._multiview_agg(human_rep, pixel_feat)
        alpha = self._alpha_forward(inter_net)
  
        ### mask based on alpha value 
        rgb = torch.zeros((alpha.shape[0], 3, alpha.shape[2])).to(alpha.device) # 1, 3, 8184
        density_mask = (alpha > 0).flatten(0,2) # 1, 1, 8184 -> 8184 # Fix bug: https://github.com/pansanity666/TransHuman/issues/3
 
        if density_mask.sum() > 0:
            inter_net = inter_net[..., density_mask]
            pixel_feat = pixel_feat[..., density_mask]
            sincos_viewdir = sincos_viewdir[:, density_mask, :]
            _rgb = self._RGB_forward(inter_net, pixel_feat, sincos_viewdir)
            rgb[:, :, density_mask] = _rgb

        ### integrate rgb and alpha 
        raw = torch.cat((rgb, alpha), dim=1) # 1, 4, 65536 
        raw = raw.transpose(1, 2) # 1, 65536, 4
        
        return raw
    
    def _multiview_agg(self, human_rep, pixel_feat):
        
        net_ske = self.actvn(self.fc_0(human_rep)) # 3, 256, 65536
        net_pix = self.actvn(self.alpha_res_0(pixel_feat)) # 3, 256, 65536

        net = self.cross_attention(net_ske, net_pix) # 3, 256, 65536
        net = self.actvn(self.fc_1(net)) 
        inter_net = self.actvn(self.fc_2(net))
        
        return inter_net

    def _alpha_forward(self, inter_net):
        opa_net = combine_interleaved(inter_net, self.n_input_holder, "average") # torch.Size([1, 256, 65536])
        opa_net = self.actvn(self.fc_3(opa_net))
        alpha = self.alpha_fc(opa_net) # 1, 1, 65536
        return alpha

    def _RGB_forward(self, inter_net, pixel_feat, sincos_viewdir):

        ## res1
        inter_features = self.feature_fc(inter_net) # inter features for color branch 
     
        features = inter_features + self.rgb_res_0(pixel_feat)           

        ## integrate sincos_viewdir 
        sincos_viewdir = repeat_interleave(sincos_viewdir, self.n_input)
        sincos_viewdir = sincos_viewdir.transpose(1, 2) # 3, 27, 65536
        features = torch.cat((features, sincos_viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        
        ## res2
        net = net + self.rgb_res_1(pixel_feat)

        ## agg
        net = combine_interleaved(net, self.n_input, "average")
            
        ## final mlp 
        net = self.actvn(self.fc_4(net))
        rgb = self.rgb_fc(net) # [1, 3, 65536]
        
        return rgb 
  
class SparseConvNet(nn.Module):
    def __init__(self, k=3):
        super(SparseConvNet, self).__init__()
  
        # NOTE: 
        self.conv0 = double_conv(cfg.embed_size, 64, 'subm0', k) # 2 SubMConv3d 
        self.down0 = stride_conv(64, 64, 'down0') # 1 SparseConv3d with stride.
         
        self.conv1 = double_conv(64, 64, 'subm1', k) # 2 SubMConv3d 
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')
    
    def forward(self, x, grid_coords):
        
        # grid_coords: [-1, 1]
        # print(x.dense().shape) # torch.Size([1, 192, 150, 150, 150])
        net = self.conv0(x) 
        # print(net.dense().shape) # torch.Size([1, 64, 150, 150, 150])
        net = self.down0(net) 
        # print(net.dense().shape)  # torch.Size([1, 64, 150, 150, 150]) 
        net = self.conv1(net)
        # print(net.dense().shape) # torch.Size([1, 64, 150, 150, 150])
        net1 = net.dense()
        # print(net.dense().shape) # torch.Size([1, 64, 150, 150, 150])
  
  
        # print(net1.shape, grid_coords.shape) # torch.Size([1, 64, 96, 176, 160]) torch.Size([1, 1, 1, 65536, 3])
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        # print(feature_1.shape) # torch.Size([1, 64, 1, 1, 65536])

        if 'ShortSparseConv' in cfg.exp_name:
            features = feature_1            
        else:
            net = self.down1(net)
            net = self.conv2(net)
            net2 = net.dense()
            feature_2 = F.grid_sample(net2,
                                      grid_coords,
                                      padding_mode='zeros',
                                      align_corners=True)
        
            net = self.down2(net)
            net = self.conv3(net)
            net3 = net.dense()
            feature_3 = F.grid_sample(net3,
                                      grid_coords,
                                      padding_mode='zeros',
                                      align_corners=True)

            net = self.down3(net)
            net = self.conv4(net)
            net4 = net.dense()
            feature_4 = F.grid_sample(net4,
                                      grid_coords,
                                      padding_mode='zeros',
                                      align_corners=True)

            features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                                 dim=1)
        
        features = features.view(features.size(0), -1, features.size(4))

        return features

def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None, k=3):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   k,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   k,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None, k=3):
    
    if 'ShortSparseConv' in cfg.exp_name:
        return SparseSequential(
            SparseConv3d(in_channels,
                        out_channels,
                        k, # kernel_size
                        #  2, # stride = 2 
                        1, # NOTE: stride, since the transformer already contains global interactions
                        padding=1,
                        bias=False,
                        indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
    
    else:
        return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     k, # kernel_size
                     2, # stride = 2 报错，未知原因。
                    #  1, # NOTE: stride, since the transformer already contains global interactions
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
        
