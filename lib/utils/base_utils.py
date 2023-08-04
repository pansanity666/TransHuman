import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F 
import math



def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    

def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        @ c2w
    )
    return c2w

def coord_from_blender(dtype=torch.float32, device="cpu"):
    """
    Blender to standard coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def coord_to_blender(dtype=torch.float32, device="cpu"):
    """
    Standard to Blender coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )
    
    
def solve_lsr(PE_can_list, PE_obs_list):
    rot_mtx_forward = []
    rot_mtx_back = []
    for xyz_can, xyz_obs in zip(PE_can_list, PE_obs_list):
        xyz_can = xyz_can.float()
        xyz_obs = xyz_obs.float()
        # print(xyz_can.shape, xyz_obs.shape) # N, 3 
        rot_mtx_forward.append(torch.linalg.lstsq(xyz_can, xyz_obs).solution) 
        rot_mtx_back.append(torch.linalg.lstsq(xyz_obs, xyz_can).solution) # 3, 3 
        # print(rot_mtx_forward.shape, rot_mtx_back.shape) 
    rot_mtx_forward = torch.stack(rot_mtx_forward) 
    rot_mtx_back = torch.stack(rot_mtx_back)
    
    return rot_mtx_forward, rot_mtx_back


def gaussion(x, u=0, rou=0.3):
  return (1 / torch.sqrt(torch.tensor(2 * math.pi * rou))) * torch.exp(-((x-u)**2)/(2*rou**2))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_quaternion(v0, v1):
    # v0: N, D (normalized)
    # v1: N, D (normalized)

    cross = torch.cross(v0, v1)  # N, 3, cross = sin_theta * unit_vecter
    axis = F.normalize(cross, dim=-1)
    sin_theta = torch.norm(cross, dim=-1) 
    cos_theta = torch.bmm(v0.unsqueeze(-2), v1.unsqueeze(-1)).squeeze() # 900, # COS(theta): cosine_similarity [-1, 1 ]
    theta = torch.atan(sin_theta / cos_theta)
    # print((sin_theta / cos_theta).min(), (sin_theta / cos_theta).max())
    
    # print(theta, theta.min(), theta.max())
    # print(torch.arcsin(sin_theta), torch.arcsin(sin_theta).min(), torch.arcsin(sin_theta).max())
    # print(torch.arccos(cos_theta), torch.arccos(cos_theta).min(), torch.arccos(cos_theta).max())
    
    sin_theta2 = torch.sin(theta/2)
    cos_theta2 = torch.cos(theta/2)
    
    # quarternion = torch.cat([axis, theta.unsqueeze(-1)], dim=-1) # 900, 4
    quarternion = torch.cat([cos_theta2.unsqueeze(-1), axis * sin_theta2.unsqueeze(-1)], dim=-1)
 
    return quarternion
    

def batch_rod2quat(rot_vecs):
    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-16, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    qx = rx * sin
    qy = ry * sin
    qz = rz * sin
    qw = cos - 1.0

    return torch.cat([qx, qy, qz, qw], dim=1)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def write_K_pose_inf(K, poses, img_root):
    K = K.copy()
    K[:2] = K[:2] * 8
    K_inf = os.path.join(img_root, 'Intrinsic.inf')
    os.system('mkdir -p {}'.format(os.path.dirname(K_inf)))
    with open(K_inf, 'w') as f:
        for i in range(len(poses)):
            f.write('%d\n'%i)
            f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(K.reshape(9).tolist()))
            f.write('\n')

    pose_inf = os.path.join(img_root, 'CamPose.inf')
    with open(pose_inf, 'w') as f:
        for pose in poses:
            pose = np.linalg.inv(pose)
            A = pose[0:3,:]
            tmp = np.concatenate([A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))



