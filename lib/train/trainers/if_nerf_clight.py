import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
from lib.train import make_optimizer

import os 
import re 
from third_parties.lpips import LPIPS

import torchvision

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

class NetworkWrapper(nn.Module):
    def __init__(self, cfg, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net)
        
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss

        self.l2rec_weight = cfg.l2rec_weight
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        print('MSE Loss initialized, weight: {}'.format(self.l2rec_weight))

        self.lpips_weight = cfg.lpips_weight
        self.lpips = LPIPS(net='vgg')
        set_requires_grad(self.lpips, requires_grad=False)
        print('Perceptual Loss initialized, weight: {}'.format(self.lpips_weight))
     
    def forward(self, batch):
        
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0
        
        if cfg.patch.use_patch_sampling:
            assert ret['rgb_map'].shape[0] == 1 
            
            ### prepare RGB 
            patch_masks = batch['patch_masks'][0] # the bbox mask patches, in

            bgcolor = torch.tensor([0.,0.,0.]).to(ret['rgb_map'][0].device)
            imgs_fake = _unpack_imgs(ret['rgb_map'][0], 
                                     patch_masks, 
                                     bgcolor=bgcolor, 
                                     targets=batch['target_patches'][0], 
                                     div_indices=batch['patch_div_indices'][0])
            targets = batch['target_patches'][0]

            ### MSE 
            mse_loss = self.l2rec_weight * self.img2mse(imgs_fake, targets)
            loss += mse_loss  
          
            ### LPIPS
            lpips_loss = self.lpips(scale_for_lpips(imgs_fake.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(targets.permute(0, 3, 1, 2)))
            lpips_loss = self.lpips_weight * torch.mean(lpips_loss)
            loss += lpips_loss
          
            scalar_stats.update({'mse_loss': mse_loss,
                                'lpips_loss': lpips_loss})
            
        else:
            mask = batch['mask_at_box']
            img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask]) 
            scalar_stats.update({'img_loss': img_loss})
            loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    
    N_patch = len(div_indices) - 1

    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
   
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs
