import os

import numpy as np
import pickle
import torch

from cv2 import Rodrigues as rodrigues

MALE_PATH    = "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
FEMALE_PATH  = "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
NEUTRAL_PATH = "SMPL_NEUTRAL.pkl"



# def transform_can_smpl(xyz):
#     center = np.array([0, 0, 0]).astype(np.float32)
#     rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
#     rot = rot.astype(np.float32)
#     trans = np.array([0, 0, 0]).astype(np.float32)
#     # if np.random.uniform() > cfg.rot_ratio:
#     #     return xyz, center, rot, trans

#     xyz = xyz.copy()

#     # rotate the smpl
#     rot_range = np.pi / 32
#     t = np.random.uniform(-rot_range, rot_range)
#     rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
#     rot = rot.astype(np.float32)
#     center = np.mean(xyz, axis=0)
#     xyz = xyz - center
#     xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
#     xyz = xyz + center

#     # translate the smpl
#     x_range = 0.05
#     z_range = 0.025
#     x_trans = np.random.uniform(-x_range, x_range)
#     z_trans = np.random.uniform(-z_range, z_range)
#     trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
#     xyz = xyz + trans

#     return xyz, center, rot, trans

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
    

class SMPL():
    def __init__(self, sex, model_dir):
        super(SMPL, self).__init__()

        model_paths = {
            'male': os.path.join(model_dir, MALE_PATH),
            'female': os.path.join(model_dir, FEMALE_PATH),
            'neutral': os.path.join(model_dir, NEUTRAL_PATH)
        }

        with open(model_paths[sex], 'rb') as f:
            smpl_model = pickle.load(f, encoding='latin1')
        self.J_regressor = np.array(smpl_model['J_regressor'].todense())
        self.weights = smpl_model['weights']
        self.posedirs = smpl_model['posedirs']
        self.v_template = smpl_model['v_template']
        self.shapedirs = np.array(smpl_model['shapedirs'])
        self.faces = smpl_model['f'].astype('int32')
        self.kintree_table = smpl_model['kintree_table'].astype('int64')
        # print(self.kintree_table)
        # print(self.posedirs.shape) # (6890, 3, 207)
        # print(self.weights.shape) # (6890, 24)

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.parent = np.array([id_to_col[self.kintree_table[0, it]] for it in range(1, self.kintree_table.shape[1])])

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)

        self.verts = None
        self.J = None
        self.R = None

    def __call__(self, pose, beta, ratio=1.0):

        if ratio == 1.0:
            return self._call(pose, beta, self.v_template)
        else:
            return self._call(pose, beta, self.v_template * ratio)
    
    def _call(self, pose, beta, v_template):
        # print(pose.shape, beta.shape, v_template.shape) # (1, 72) (10,) (6890, 3)
            
        # v_template = self.v_template              # (6890, 3)
        shapedirs = self.shapedirs.reshape(-1,10) # (6890, 10)
        beta = beta[:, None]                      # (10, 1)

        v_shaped = shapedirs.dot(beta).reshape(6890, 3) + v_template # (6890, 3)
        # write_point_cloud('./v_shaped.ply', v_shaped)
        # print(v_shaped)
        
        # the joints of new shape
        J = self.J_regressor.dot(v_shaped)                           # (24, 3)

        # input is a rotation matrix: (24,3,3)
        if pose.shape == (24, 3, 3):
            R = pose
        # input is a rotation axis-angle vector: (1, 72), (72, 1) or (72, )
        elif pose.shape == (1, 72) or pose.shape == (72, 1) or pose.shape == (72,):
            pose_vectors = pose.reshape(-1, 3)                      # (24, 3)
            R = np.array([rodrigues(pose_vectors[p_idx])[0] 
                            for p_idx in range(pose_vectors.shape[0])
                          ], 
                          dtype='float32')                          # (24, 3, 3)
        else:
            raise ValueError("Unsupported Pose Inputs - the Pose Shape is {}".format(pose.shape))
        # print(R.shape) # 24, 3, 3
        # print(R[0])
        # print(R[1])
    
        Is = np.eye(3, dtype='float32')[None, :]                    # (1, 3, 3)
        lrotmin = (R[1:,:] - Is).reshape(-1, 1)                     # (23x3x3, 1)
        posedirs = self.posedirs.reshape(-1,207)                    # (6890, 207)
        v_posed = v_shaped + posedirs.dot(lrotmin).reshape(6890, 3) # (6890, 3)
        # write_point_cloud('./v_posed.ply', v_posed)
        
        J_ = J.copy()
        J_[1:, :] = J[1:, :] - J[self.parent, :]                     # (24, 3)
        G_ = np.concatenate([R, J_[:, :, None]],  axis=-1)           # (24, 3, 4)
        pad_rows = np.array([[0, 0, 0, 1]], dtype='float32')
        pad_rows = np.repeat(pad_rows, 24, axis=0).reshape(-1, 1, 4)
        G_ = np.concatenate([G_, pad_rows], axis=1)                  # (24, 4, 4)

        G = [G_[0].copy()]
        for i in range(1, 24):
            G.append(G[self.parent[i-1]].dot(G_[i, :, :]))
        G = np.stack(G, axis=0)  # (24, 4, 4)

        joints = G[:, :3, 3]

        rest_joints = np.concatenate([J, np.zeros((24, 1))], axis=-1)[:, :, None]  # (24, 4, 1)
        zeros = np.zeros((24, 4, 3), dtype='float32')                              # (24, 4, 3)
        rest_joints_mtx = np.concatenate([zeros, rest_joints], axis=-1)            # (24, 4, 4) 
        posed_joints_mtx = np.matmul(G, rest_joints_mtx)
        G = G - posed_joints_mtx
                                                                    
        # print(G.shape) # 24, 4, 4
        # print(v_posed)
        # is v_poses == canonical smpl ?
        rest_shape_h = np.concatenate([v_posed, np.ones(v_posed.shape[0])[:, None]], axis=-1) #(6890, 4)
        T = self.weights.dot(G.reshape(24, -1)).reshape(6890, 4, 4) # (6890, 24) x (24, 4, 4) -> (6890, 4, 4)        
        v = np.matmul(T, rest_shape_h[:, :, None]) # (6890, 4, 4) x (6890, 4, 1) -> 6890, 4, 1 
        # print(v.shape)
        # write_point_cloud('./v.ply', v[:, :3, 0])
        
        # inv_T = torch.inverse(torch.tensor(T)).cpu().numpy()
        # ori_v = np.matmul(inv_T, v)
        # write_point_cloud('./ori_v.ply', ori_v[:, :3, 0])
        # assert False
        
        v = v[:, :3, 0]
        
        return v, joints, T 