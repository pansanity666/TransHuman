import numpy as np
import json
import os
import cv2
import torch

from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.base_utils import euclidean_dist
from pytorch3d.ops import knn_points


def batch_index(batch_tensor, batch_idx):
    # batch_tensor: B, N, xxx
    # batch_idx: B, N_idx
    
    r = []
    for t, i in zip(batch_tensor, batch_idx):
        r.append(t[i])
    
    return torch.stack(r)
    
def LBS_mapping(pts, smpl_source, smpl_tar, blend_mtx_source, blend_mtx_tar):
    ### Transform pts from source to tar space using INV LBS. (All under SMPL coordinate.)
    # pts: B, N, 3
    # smpl_obs: B, 6890, 3 
    # smpl_can: B, 6890, 3 
    # blend_mtx_source: B, 6890, 4, 4 
    # blend_mtx_tar: B, 6890, 4, 4 
    
    # find nearest SMPL vertex for each pt
    dist, idx, Vnn = knn_points(pts, smpl_source, K=1, return_nn=True) # torch.Size([1, 65536, 1]) torch.Size([1, 65536, 1]) torch.Size([1, 65536, 1, 3])
    blend_mtx_source_nearest = batch_index(blend_mtx_source, idx[..., 0]) # B, N, 4, 4 
    inv_blend_mtx_source_nearest = torch.inverse(blend_mtx_source_nearest) # B, N, 4, 4 
    blend_mtx_tar_nearest = batch_index(blend_mtx_tar, idx[..., 0]) # B, N, 4, 4 
    
    # 1. From source to canonical.
    pts = torch.cat([pts, torch.ones_like(pts)[...,:1]], dim=-1) # B, N, 4
    pts_can = torch.matmul(inv_blend_mtx_source_nearest, pts.unsqueeze(-1).to(torch.double)) # B, N, 4, 1 
    
    # 2. From canonical to target. 
    pts_tar = torch.matmul(blend_mtx_tar_nearest, pts_can)[..., :3, 0]
    
    return pts_tar

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


def barycentric_mapping(pts, smpl_obs, smpl_can, face_idx):
    # pts: B, N, 3
    # smpl_obs: B, 6890, 3 
    # smpl_can: B, 6890, 3 

    B = pts.shape[0]
    meshes = smpl_obs[:, face_idx] # # 1, 13776, 3, 3
    closest_meshes, idx = get_closest_mesh(pts, meshes) # 

    # print(idx.shape) # torch.Size([1, 65536, 1]) # the index of 13776 meshes.
    # print(closest_meshes.shape, pts.shape) 
    # torch.Size([1, 65536, 3, 3]) torch.Size([1, 65536, 3])
    
    # write_point_cloud('./CAN_pts.ply', torch.cat([smpl_can[0], pts[0,:64]], dim=0).cpu().numpy() )
    # write_point_cloud('./OBS_pts.ply', torch.cat([smpl_obs[0], pts[0,:64]], dim=0).cpu().numpy() )
    # write_point_cloud('./BM_pts.ply', torch.cat([pts[0,:64], closest_meshes[0,:64].flatten(0,1)], dim=0).cpu().numpy() )

    uv, signed_distance = project_point2mesh(
        pts.reshape(-1,3), meshes=closest_meshes.reshape(-1,3,3)
    )
    # print(uv.shape, signed_distance.shape) # torch.Size([65536, 2]) torch.Size([65536])
    transparent_mask = get_transparent_mask(uv, signed_distance).reshape(B,-1)
    # print(transparent_mask.shape) # torch.Size([1, 65536])

    # print(smpl_can.shape) # 1, 6890, 3
    meshes_can = smpl_can[:,face_idx]
    # print(meshes_can.shape) # torch.Size([1, 13776, 3, 3])
    meshes_can = meshes_can[0][idx.flatten()] # cloeset canonical mesh for each point. 
    # print(meshes_can.shape) # 65536, 3, 3
    # write_point_cloud('./CAN_pts_meshes.ply', torch.cat([pts[0,:64], meshes_can[:64].flatten(0,1)], dim=0).cpu().numpy() )
 
    pts_smpl_can = barycentric_map2can(uv, signed_distance, meshes_can)
    # write_point_cloud('./CAN_pts_meshes.ply', torch.cat([pts_smpl_can[:64], pts[0,:64], meshes_can[:64].flatten(0,1)], dim=0).cpu().numpy() )
    # write_point_cloud('./CAN_pts.ply', torch.cat([smpl_can[0], pts_smpl_can[:64]], dim=0).cpu().numpy() )

    return pts_smpl_can, meshes_can, idx.flatten()

def barycentric_map2can(uv, signed_distance, meshes_can):
    """map points to canonical space by "uv and distance" coordinate

    Args:
        uv ([type]): [description]
        signed_distance ([type]): [description]
        meshes_can ([type]): [description]

    Returns:
        [type]: [description]
    """
    v2 = meshes_can[..., 2, :] - meshes_can[..., 0, :]
    v1 = meshes_can[..., 1, :] - meshes_can[..., 0, :]
    normal_f = torch.cross(v1, v2)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    offset_vec = signed_distance.unsqueeze(-1) * normal_f
    pts_proj_can = meshes_can[..., 0, :] + uv[:, 0, None] * v2 + uv[:, 1, None] * v1
    pts_smpl_can = pts_proj_can + offset_vec
    return pts_smpl_can


def get_transparent_mask(uv, signed_distance, floor=-4, ceil=5, max_dist=0.1):
 
    # print(uv.shape) # torch.Size([352000, 2])
    clamped_uv_mask = torch.logical_or(uv > ceil, uv < floor)
    # print(clamped_uv_mask.shape) # torch.Size([352000, 2])

    # mask each point which has uv in [-inf, -4] or [5, +inf] or signed_distance > 0.1 
    transparent_mask = torch.logical_or(clamped_uv_mask[:, 0], clamped_uv_mask[:, 1]) 
    transparent_mask = torch.logical_or(
        transparent_mask, signed_distance.abs() > max_dist 
    )
    
    return transparent_mask

def get_barycentric_coordinates(pts_proj, meshes):
    v0 = meshes[..., 2, :] - meshes[..., 0, :]
    v1 = meshes[..., 1, :] - meshes[..., 0, :]
    v2 = pts_proj - meshes[..., 0, :]

    dot00 = (v0 * v0).sum(-1)
    dot01 = (v0 * v1).sum(-1)
    dot02 = (v0 * v2).sum(-1)
    dot11 = (v1 * v1).sum(-1)
    dot12 = (v1 * v2).sum(-1)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    uv = torch.stack([u, v], dim=-1)
    return uv


def project_point2mesh(pts, meshes):
    """project points to corresponding meshes,points number must be the same as mesh number

    Args:
        pts (tensor): [n,3]
        meshes (tensor): [n,3,3]
    """
    assert (
        pts.shape[0] == meshes.shape[0]
    ), "points number must be the same as mesh number"
    v10 = meshes[:, 1] - meshes[:, 0]
    v20 = meshes[:, 2] - meshes[:, 0]
    normal_f = torch.cross(v10, v20)
    normal_f = normal_f / torch.norm(normal_f, dim=-1, keepdim=True)
    tmp = pts - meshes[:, 0]
    signed_distance = torch.einsum("ij,ij->i", tmp, normal_f)
    pts_proj = pts - normal_f * signed_distance.unsqueeze(-1)
    uv = get_barycentric_coordinates(pts_proj=pts_proj, meshes=meshes)

    return uv, signed_distance

def get_closest_mesh(vsrc, meshes):
    """get closest mesh by barycentric points of each mesh

    Args:
        vsrc ([type]): [description]
        meshes ([type]): [description]

    Returns:
        [type]: [description]
    """
    mesh_centroid = meshes.mean(dim=-2)
    # print(vsrc.shape, meshes.shape) torch.Size([1, 352000, 3]) torch.Size([1, 13776, 3, 3])
    
    # print(vsrc.shape, mesh_centroid.shape) # torch.Size([1, 65536, 3]) torch.Size([1, 13776, 3])
    dist, idx, Vnn = knn_points(vsrc, mesh_centroid, K=1, return_nn=True)

    ### the self implemented euclidean distance leads to OOM in 32G V100.
    # assert vsrc.shape[0] == 1 # only support bs = 1 ATM. 
    # dist = euclidean_dist(vsrc.flatten(0,1), mesh_centroid.flatten(0,1))
    # # print(dist.sqrt())
    # _, idx = torch.argsort(dist, dim=1)[:, :1] 
    # print(idx)
    # assert False

    # print(dist.shape) # torch.Size([1, 352000, 1])
    # print(idx.shape) # torch.Size([1, 352000, 1])
    # print(Vnn.shape) # torch.Size([1, 352000, 1, 3])

    closest_meshes = torch.gather(
        meshes, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 3, 3)
    )
    # print(closest_meshes.shape) # torch.Size([1, 352000, 3, 3])

    return closest_meshes, idx

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def load_cam(ann_file):
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
        K[i][:2] = K[i][:2] * cfg.ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT


def get_center_rayd(K, RT):
    H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
    RT = np.array(RT)
    ray_o, ray_d = if_nerf_dutils.get_rays(H, W, K,
                            RT[:3, :3], RT[:3, 3])
    return ray_d[H // 2, W // 2]


def gen_path(RT, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, cfg.render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c

def gen_path_virt(RT, center=None, render_views=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []

    for theta in np.linspace(0., 2 * np.pi, render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c

def read_voxel(frame, args):
    voxel_path = os.path.join(args['data_root'], 'voxel', args['human'],
                              '{}.npz'.format(frame))
    voxel_data = np.load(voxel_path)
    occupancy = np.unpackbits(voxel_data['compressed_occupancies'])
    occupancy = occupancy.reshape(cfg.res, cfg.res,
                                  cfg.res).astype(np.float32)
    bounds = voxel_data['bounds'].astype(np.float32)
    return occupancy, bounds


def image_rays(RT, K, bounds):
    H = cfg.H * cfg.ratio
    W = cfg.W * cfg.ratio
    ray_o, ray_d = if_nerf_dutils.get_rays(H, W, K,
                            RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = if_nerf_dutils.get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    return ray_o, ray_d, near, far, center, scale, mask_at_box


def get_image_rays0(RT0, RT, K, bounds):
    """
    Use RT to get the mask_at_box and fill this region with rays emitted from view RT0
    """
    H = cfg.H * cfg.ratio
    ray_o, ray_d = if_nerf_dutils.get_rays(H, H, K,
                            RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = if_nerf_dutils.get_near_far(bounds, ray_o, ray_d)

    ray_o, ray_d = if_nerf_dutils.get_rays(H, H, K,
                            RT0[:3, :3], RT0[:3, 3])
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d[mask_at_box]

    return ray_d


def save_img(img, frame_root, index, mask_at_box):
    H = int(cfg.H * cfg.ratio)
    rgb_pred = img['rgb_map'][0].detach().cpu().numpy()
    mask_at_box = mask_at_box.reshape(H, H)

    img_pred = np.zeros((H, H, 3))
    img_pred[mask_at_box] = rgb_pred
    img_pred[:, :, [0, 1, 2]] = img_pred[:, :, [2, 1, 0]]

    print("saved frame %d" % index)
    cv2.imwrite(os.path.join(frame_root, '%d.jpg' % index), img_pred * 255)
