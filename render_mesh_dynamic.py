import trimesh 
# from lib.datasets import get_human_info
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import chumpy
import torch
from torch import nn
import os 
from tqdm import tqdm

import open3d as o3d

from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply 

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    SfMPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection

from pytorch3d.renderer.blending import BlendParams

import torchvision
from torchvision import transforms
import imageio
import cv2
import sys
import warnings
warnings.filterwarnings("ignore")

from lib.utils import render_utils

import json

from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)

image_to_tensor = transforms.ToTensor()

human_info = {
            'CoreView_313': {'begin_i': 60, 'i_intv': 1, 'ni': 1000},
            'CoreView_315': {'begin_i': 400, 'i_intv': 1, 'ni': 1000},
            'CoreView_377': {'begin_i': 300, 'i_intv': 1, 'ni': 317},
            'CoreView_386': {'begin_i': 300, 'i_intv': 1, 'ni': 346},
            'CoreView_390': {'begin_i': 0, 'i_intv': 1, 'ni': 700},
            'CoreView_392': {'begin_i': 300, 'i_intv': 1, 'ni': 256},
            'CoreView_396': {'begin_i': 1080, 'i_intv': 1, 'ni': 270},
            'CoreView_387': {'begin_i': 0, 'i_intv': 1, 'ni': 654}, # 0-300
            'CoreView_393': {'begin_i': 0, 'i_intv': 1, 'ni': 658},  # 0-300 
            'CoreView_394': {'begin_i': 0, 'i_intv': 1, 'ni': 859}  # 0-300
            }

def process_rendered(rendered_img, K_opencv, D_opencv):

    # input: 1024, 1024, 3
    rendered_img = cv2.undistort(rendered_img, K_opencv, D_opencv)
    rendered_img = cv2.resize(rendered_img, (512, 512),
                            interpolation=cv2.INTER_AREA)
    rendered_img = image_to_tensor(rendered_img)

    return rendered_img

def write_point_cloud(ply_filename, points):

    if points.shape[1] == 3: 
        colors = np.zeros((points.shape[0],3))
        points = np.concatenate((points, colors), axis=1)

    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def get_smpl_vertice(human, frame):
        
    if '396' in human:
        vertices_path = os.path.join('./data/zju_mocap', human, 'vertices',
                                    '{}.npy'.format(frame))
    else:
        vertices_path = os.path.join('./data/zju_mocap', human, 'new_vertices',
                                    '{}.npy'.format(frame))
    smpl_vertice = np.load(vertices_path).astype(np.float32)

    return smpl_vertice

def get_can_bounds(human, i):
    
    xyz = get_smpl_vertice(human, i)
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    can_bounds = np.stack([min_xyz, max_xyz], axis=0)
    
    return can_bounds

def get_params(human, i):
    
    if '396' in human:
        params_path = os.path.join('./data/zju_mocap', human, 'params', '{}.npy'.format(i))
    else:
        params_path = os.path.join('./data/zju_mocap', human, 'new_params', '{}.npy'.format(i))
    params = np.load(params_path, allow_pickle=True).item()

    return params

def rotate_mesh_smpl(verts, human, i):
    # from world to smpl coordinate 
    params = get_params(human, i)
    Rh = params['Rh'] # 1, 3 
    Rh = cv2.Rodrigues(Rh)[0].astype(np.float32) # 3, 3 # convert rotation vector to rotation matrix
    Th = params['Th'].astype(np.float32)
    verts = np.dot(verts - Th, Rh)
    
    return verts
    
def rotate_mesh_camera(verts, R, T):
  
    verts = torch.from_numpy(verts)
    R = torch.from_numpy(R)
    T = torch.from_numpy(T)
    
    verts = torch.matmul(R.unsqueeze(0), verts.unsqueeze(-1))[...,0] + T
    
    # inverse z and y aixs 
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html
    # z (blue) out, y (green) up, x (red) right 
    verts[...,-1] *= -1
    verts[...,-2] *= -1
 
    return verts.numpy()

def render_depth_map(verts, faces, can_bounds, R, T, K, human, i):
    # verts: 6890, 3, tensor 
    # R: b, 3, 3, tensor, 
    # T: b, 3,  tensor
    # K: b, 3, 3, tensor 

    ### paint mesh with normal map from camera coordinate system 
    mesh = Meshes([verts], [faces], ).to(device)
    verts_norm = rotate_mesh_camera(verts.cpu().numpy(), R[0].cpu().numpy(), T[:1].cpu().numpy())
    verts_norm = torch.from_numpy(verts_norm)
    mesh_norm = Meshes(
                    [verts_norm], [faces],
                ).to(device)
    # default ratio should be 0.5, we use higher value here for better brightness
    normals = 0.7*mesh_norm.verts_normals_packed()[None].to(device) + 0.7

    mesh.textures = Textures(verts_rgb=normals)

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 

    fx, fy = K[:,0,:1], K[:,1,1:2]
    px, py = K[:,0,2:], K[:,1,2:]
    focals = torch.cat([fx, fy], dim=-1)
    prins = torch.cat([px, py], dim=-1)
    
    cameras = cameras_from_opencv_projection(R, T, K, image_size=torch.tensor([[512,512]]))

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    blend_params = BlendParams(
        background_color=[1,1,1]
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 3.0, 0.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )

    # rendering 
    images, depth = renderer(mesh)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.savefig('./render_vis_test.jpg')
    # plt.cla()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(depth[0].cpu().numpy(), cmap=plt.cm.gray_r)
    # plt.axis("off")
    # plt.savefig('./render_depth_test.jpg')

    return images[..., :3], depth

def load_cam(ann_file, ratio=1.0):
    
    if ann_file.endswith('.json'):
        annots = json.load(open(ann_file, 'r'))
        cams = annots['cams']['20190823']
    else:
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']

    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        K[i][:2] = K[i][:2] * ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT


if __name__=='__main__':

    device = 'cuda:0'
    ratio = 0.5

### NOTE: Set this two parameters as you need. 
human = 'CoreView_313'
DATA_PATH = './data/mesh/official/epoch_2100/model_o_motion_x/' + human

# get human split info for current human 
info = human_info[human]
begin_i, i_intv, ni = info['begin_i'], info['i_intv'], info['ni']

# init dynamic cameras for current human 
ann_file = os.path.join('./data/zju_mocap/', human, 'annots.npy')
K, RT = load_cam(ann_file, ratio=ratio)
render_w2c = render_utils.gen_path_virt(RT, render_views=(ni))

# 
for i in tqdm(range(begin_i, begin_i+ni)):
    
    ### 313 and 315 frame index starts from 1 
    if human in ['CoreView_313', 'CoreView_315']: i += 1 

    ## load vertice
    v = DATA_PATH + '/{}.ply'.format(i) 
    print('Loading vertice from ', v)
    verts, faces = load_ply(v)
    
    ## get can_bounds 
    can_bounds = get_can_bounds(human, i)

    cam_ind = i % len(render_w2c)
    RT = render_w2c[cam_ind]
    R_opencv, T_opencv = RT[:3, :3], RT[:3, 3:]
    K_opencv = K[0]

    K_opencv = np.array(K_opencv)
    R_opencv = np.array(R_opencv)
    T_opencv = np.array(T_opencv) 

    K_opencv = torch.tensor(K_opencv).unsqueeze(0).float().to(device)
    R_opencv = torch.tensor(R_opencv).unsqueeze(0).float().to(device)
    T_opencv = torch.tensor(T_opencv).unsqueeze(0).float().squeeze(-1).to(device)
    
    # render 1024x then resize to 512x 
    rendered_img, rendered_depth = render_depth_map(verts, faces, can_bounds, R_opencv, T_opencv, K_opencv, human, i)

    save_path = DATA_PATH + '_dyn/{}.jpg'.format(i)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchvision.utils.save_image(rendered_img.permute(0,3,1,2), save_path)
    