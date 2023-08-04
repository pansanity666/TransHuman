"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
import functools
from lib.config import cfg
import numpy as np
import time
from lib.networks.vision_transformer import PositionalEncoding


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm_type)
    return norm_layer

def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net

class SpatialEncoder(nn.Module):
    def __init__(
            self,
            backbone="resnet18",
            pretrained=True,
            num_layers=3,  # 2: 128 3: 256 4: 512
            index_interp="bilinear",
            index_padding="zeros",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
    ):
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        # print(backbone) # resnet18
        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        pretrained = cfg.pretrained
        print("Using torchvision", backbone, "encoder")
        print('Pretrained: ' + str(pretrained))
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )

        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        
        # self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
        self.reduction_layer = nn.Conv2d(cfg.img_feat_size+128, cfg.embed_size,1)
        
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

        # Please ignore this line.
        self.PE_color = PositionalEncoding(num_freqs=10)
        
        self.upsample_color = nn.Conv2d(3, 128, 1)
        
    def forward(self, x):
        # x: b, c, h, w 
        ori_H, ori_W = x.shape[2:]
        x_ori = x
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )

        pixel_feat_map = None
        holder_feat_map = None
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x) 
            latents.append(x) 
        if self.num_layers > 2:
            x = self.model.layer2(x)  
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)
     
        # interpolate feature map to original size
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                (ori_H, ori_W),
                mode=self.upsample_interp,
                align_corners=None if self.index_interp == "nearest " else True,
            )
        pixel_feat_map = torch.cat(latents, dim=1)

        # fine-grained appearance feature integration
        x_ori = self.upsample_color(x_ori)
        pixel_feat_map = torch.cat([pixel_feat_map, x_ori], dim=1)  # 3, 63 + 256, 512, 512
        holder_feat_map = self.reduction_layer(pixel_feat_map)
        
        # salces used in projection 
        pixel_feat_scale = np.array([pixel_feat_map.shape[-1], pixel_feat_map.shape[-2]])
        pixel_feat_scale = pixel_feat_scale / (pixel_feat_scale - 1) * 2.0
        
        holder_feat_scale = np.array([holder_feat_map.shape[-1], holder_feat_map.shape[-2]])
        holder_feat_scale = holder_feat_scale / (holder_feat_scale - 1) * 2.0
      
        return holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):

        super().__init__()
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):

        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        
        
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )


