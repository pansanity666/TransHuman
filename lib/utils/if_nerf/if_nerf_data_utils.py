import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh
import matplotlib.pyplot as plt

import torch
import torchvision

def get_rays(H, W, K, R, T):
    # get rays under world coordinate. 

    rays_o = -np.dot(R.T, T).ravel()
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    # print(xy1.shape) 512, 512, 3
    # print(K) 
    # print(np.dot(R, R.T))
    # print(np.linalg.inv(K))
    # print(np.linalg.inv(K).T)
  
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T) # pixel under camera coord
    pixel_world = np.dot(pixel_camera - T.ravel(), R) # 
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)

    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z], # 
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    # project box corners under world coord to image plane, and mask the coverd region. 
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose) # project corners from world coord to image plane? 
    corners_2d = np.round(corners_2d).astype(int) # 8, 2

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def sample_ray_grid(img, msk, K, R, T, bounds, nrays, split):

    # bounds is under workd coord

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
  
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio # 0.0
        body_sample_ratio = cfg.body_sample_ratio # 0.5
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        # sample rays on body

        human_fg = (msk != 0).astype(int)
        human_bg = (1 - (
                    (bound_mask == 1).astype(int) * (msk != 0).astype(int))) * (
                       (bound_mask == 1).astype(int))
        human_fg_idx = np.argwhere(human_fg)
        human_bg_idx = np.argwhere(human_bg)
        human_idx = np.concatenate([human_fg_idx, human_bg_idx], axis=0)

        prob_list = [cfg.sample_fg_ratio / len(human_fg_idx)] * len(
            human_fg_idx) + [
                        (1 - cfg.sample_fg_ratio) / len(human_bg_idx)] * len(
            human_bg_idx)
        sample_idx = np.random.choice(np.arange(len(human_idx)), 1, p=prob_list)
        sample_center = human_idx[sample_idx]
        h, w = int(nrays ** 0.5), int(nrays ** 0.5)

        def gen_grid(h, w):
            x_ = np.arange(0, h)
            y_ = np.arange(0, w)
            x, y = np.meshgrid(x_, y_, indexing='ij')
            grid = np.stack([x, y], axis=-1).reshape(-1, 2)
            center = -1 * np.array([h // 2, w // 2]).astype(grid.dtype)
            grid += center
            return grid

        coord = sample_center + gen_grid(h, w)

        # move the patch inside the image
        img_h, img_w = img.shape[:2]
        border = np.max([0 - coord[:, 0], coord[:, 0] - (img_h - 1)])
        border = max(border, 0)
        coord[:, 0] = coord[:, 0] - border
        border = np.max([0 - coord[:, 1], coord[:, 1] - (img_w - 1)])
        border = max(border, 0)
        coord[:, 1] = coord[:, 1] - border

        ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
        ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
        rgb_ = img[coord[:, 0], coord[:, 1]]

        near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

        def pad(input):
            pad_array = np.zeros(nrays, )
            pad_array[:len(input)] = input
            return pad_array

        near_ = pad(near_)
        far_ = pad(far_)

        ray_o_list.append(ray_o_)
        ray_d_list.append(ray_d_)
        rgb_list.append(rgb_)
        near_list.append(near_)
        far_list.append(far_)
        coord_list.append(coord)
        mask_at_box_list.append(mask_at_box)
        nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def _get_patch_ray_indices(
        ray_mask,  # bbox mask 
        subject_mask, 
        candidate_mask, 
        patch_size, 
        H, W):

    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == np.bool
    assert candidate_mask.dtype == np.bool

    # candidate_mask only determines the patch center. 
    valid_ys, valid_xs = np.where(candidate_mask)

    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0], size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    half_patch_size = patch_size // 2
    x_min = np.clip(a=center_x-half_patch_size, 
                    a_min=0, 
                    a_max=W-patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y-half_patch_size,
                    a_min=0,
                    a_max=H-patch_size)
    y_max = y_min + patch_size

    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    # # print(sel_ray_mask)
    # torchvision.utils.save_image(torch.from_numpy(sel_ray_mask).float().unsqueeze(0).unsqueeze(0), './sel_ray_mask.jpg')
    # torchvision.utils.save_image(torch.from_numpy(ray_mask.reshape(H,W)).float().unsqueeze(0).unsqueeze(0), './ray_mask.jpg')
    
    sel_ray_mask = sel_ray_mask.reshape(-1) # select pacth mask 
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask) # intersection between sampled patch and bbox ray mask 
    select_masked_inds = np.where(inter_mask)
    
    inter_mask_sub = np.bitwise_and(sel_ray_mask, subject_mask.reshape(-1)) # patch mask bitwise subject mask 
    
    # torchvision.utils.save_image(torch.from_numpy(inter_mask.reshape(H,W)).float().unsqueeze(0).unsqueeze(0), './inter_mask.jpg')
        
    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds] # indexs under valid ray area. 
    
    inter_mask = inter_mask.reshape(H, W)
    inter_mask_sub = inter_mask_sub.reshape(H, W)
    
    return select_inds, \
            inter_mask[y_min:y_max, x_min:x_max], \
            inter_mask_sub[y_min:y_max, x_min:x_max], \
            np.array([x_min, y_min]), np.array([x_max, y_max])
            
def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far
    
def get_patch_ray_indices(
        N_patch, 
        ray_mask, 
        subject_mask, 
        body_mask,
        bbox_mask,
        patch_size, 
        H, W):

    assert subject_mask.dtype == np.bool
    assert bbox_mask.dtype == np.bool

    # background in bbox 
    bbox_exclude_subject_mask = np.bitwise_and(
        bbox_mask,
        np.bitwise_not(subject_mask)
    )
    
    list_ray_indices = []
    list_mask = []
    list_mask_sub = []
    list_xy_min = []
    list_xy_max = []

    total_rays = 0
    patch_div_indices = [total_rays]
    for _ in range(N_patch):
        
        if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
            candidate_mask = subject_mask
        else:
            candidate_mask = bbox_exclude_subject_mask

        ray_indices, mask, mask_sub, xy_min, xy_max = \
            _get_patch_ray_indices(ray_mask, subject_mask, candidate_mask, patch_size, H, W)

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)

        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_mask_sub.append(mask_sub)
        list_xy_min.append(xy_min) # min under full image size 
        list_xy_max.append(xy_max) # max under full image size 
        
        patch_div_indices.append(total_rays) # the division indices of each patch 

    select_inds = np.concatenate(list_ray_indices, axis=0) # 2400 

    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'mask_sub': np.stack(list_mask_sub, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }

    patch_div_indices = np.array(patch_div_indices)

    return select_inds, patch_info, patch_div_indices

def sample_patch_rays(img, H, W,
                        subject_mask, body_mask, bbox_mask, ray_mask,
                        rays_o, rays_d, ray_img, sub_mask_in_bbox, near, far):

    select_inds, patch_info, patch_div_indices = \
        get_patch_ray_indices(
            N_patch=cfg.patch.N_patches, 
            ray_mask=ray_mask, 
            subject_mask=subject_mask, 
            body_mask=body_mask, 
            bbox_mask=bbox_mask,
            patch_size=cfg.patch.size, 
            H=H, W=W)

    rays_o, rays_d, ray_img, near, far = select_rays(
        select_inds, rays_o, rays_d, ray_img, near, far)
    sub_mask_in_bbox = sub_mask_in_bbox[select_inds]
    
    targets = []
    for i in range(cfg.patch.N_patches):
        x_min, y_min = patch_info['xy_min'][i] 
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(img[y_min:y_max, x_min:x_max])
    target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

    patch_masks = patch_info['mask']  # boolean array (N_patches, P, P), the foreground mask of size PxP
    patch_masks_sub = patch_info['mask_sub']
    return rays_o, rays_d, ray_img, sub_mask_in_bbox, near, far, \
            target_patches, patch_masks, patch_masks_sub, patch_div_indices
        
def sample_ray_patch(img, msk, K, R, T, bounds, nrays, split):
    # sample rays constrained by mask and 3D bounding boxes.
    # bounds: 3D bounds on SMPL space under world coord

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask # intersection is reserved
  
    bound_mask[msk == 100] = 0 

    if split == 'train':
    
        human_mask = msk > 0 # 512, 512  # include silhouette,

        body_mask = msk == 1
        
        ray_img = img.reshape(-1, 3)
        ray_o = ray_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        ray_d = ray_d.reshape(-1, 3)
        sub_mask = human_mask.reshape(-1, 1) # the target of mask loss, exclude silhouette  
        
        near, far, ray_mask = get_near_far(bounds, ray_o, ray_d)
    
        ray_img = ray_img[ray_mask]
        ray_o = ray_o[ray_mask]
        ray_d = ray_d[ray_mask]
        sub_mask = sub_mask[ray_mask]
        
        rays_o, rays_d, ray_img, sub_mask, near, far, \
        target_patches, patch_masks, patch_masks_sub, patch_div_indices = \
            sample_patch_rays(img=img, H=H, W=W,
                        subject_mask=human_mask, # (H,W)
                        body_mask=body_mask, 
                        bbox_mask=ray_mask.reshape(H,W), # (H,W)
                        ray_mask=ray_mask, # H*W 
                        rays_o=ray_o, 
                        rays_d=ray_d, 
                        ray_img=ray_img, 
                        sub_mask_in_bbox=sub_mask, 
                        near=near, 
                        far=far)
        
        patch_meta = {
                    'sub_mask': sub_mask, 
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'patch_masks_sub': patch_masks_sub,  # 6, 20, 20, 
                    'target_patches': target_patches
                    }
        
        return ray_img.astype(np.float32), rays_o.astype(np.float32), rays_d.astype(np.float32), near.astype(np.float32), far.astype(np.float32), patch_meta

    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
  
    # sample rays constrained by mask and 3D bounding boxes.
    # bounds: 3D bounds on SMPL space under world coord

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask # intersection is reserved

    # exclude SIL 
    bound_mask[msk == 100] = 0

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio # 0. 
        body_sample_ratio = cfg.body_sample_ratio # 0.5

        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk == 1)

            if len(coord_body) > 0:
                coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                          n_body)]
            else:
                coord_body = np.ones((1, 2), dtype=np.uint8)

            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            if len(coord) > 0:
                coord = coord[np.random.randint(0, len(coord), n_rand)]
            else:
                coord = np.zeros((1, 2), dtype=np.uint8)

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)

    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        # print(mask_at_box.shape) # 313344
     
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_smpl_data(ply_path):
    ply = trimesh.load(ply_path)
    xyz = np.array(ply.vertices)
    nxyz = np.array(ply.vertex_normals)

    if cfg.add_pointcloud:
        # add random points
        xyz_, ind_ = trimesh.sample.sample_surface_even(ply, 5000)
        nxyz_ = ply.face_normals[ind_]
        xyz = np.concatenate([xyz, xyz_], axis=0)
        nxyz = np.concatenate([nxyz, nxyz_], axis=0)

    xyz = xyz.astype(np.float32)
    nxyz = nxyz.astype(np.float32)

    return xyz, nxyz


def get_acc(coord, msk):
    border = 25
    kernel = np.ones((border, border), np.uint8)
    msk = cv2.dilate(msk.copy(), kernel)
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def rotate_smpl(xyz, nxyz, t):
    """
    t: rotation angle
    """
    xyz = xyz.copy()
    nxyz = nxyz.copy()
    center = (np.min(xyz, axis=0) + np.max(xyz, axis=0)) / 2
    xyz = xyz - center
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R = R.astype(np.float32)
    xyz[:, :2] = np.dot(xyz[:, :2], R.T)
    xyz = xyz + center

    return xyz, nxyz, center


def transform_can_smpl(xyz):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > cfg.rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans


def sample_ray_ohem(img, loss_img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = 0
        body_sample_ratio = 0
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            loss = loss_img[coord[:, 0], coord[:, 1]]
            loss = loss / loss.sum()
            n_rand0 = int(n_rand * 0.8)
            ind = np.random.choice(np.arange(len(coord)),
                                   n_rand0,
                                   replace=False,
                                   p=loss)
            coord0 = coord[ind]
            coord1 = coord[np.random.randint(0, len(coord), n_rand - n_rand0)]
            coord = np.concatenate([coord0, coord1], axis=0)

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box)
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box

def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d

def sample_smpl_ray(img, msk, depth, K, R, T, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    dpt_msk = (depth != 0).astype(np.uint8)
    pts3d = unproject(depth, K, R, T)

    if split == 'train':
        body_sample_ratio = cfg.body_sample_ratio
        face_sample_ratio = cfg.face_sample_ratio

        n_body = int(nrays * body_sample_ratio)
        n_face = int(nrays * face_sample_ratio)
        n_rand = nrays - n_body - n_face

        # sample rays on body
        coord_body = np.argwhere((msk * dpt_msk) != 0)
        coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                  n_body)]
        # sample rays on face
        coord_face = np.argwhere(msk == 13)
        if len(coord_face) > 0:
            coord_face = coord_face[np.random.randint(
                0, len(coord_face), n_face)]
        # sample rays in the depth mask
        coord = np.argwhere(dpt_msk == 1)
        coord = coord[np.random.randint(0, len(coord), n_rand)]

        if len(coord_face) > 0:
            coord = np.concatenate([coord_body, coord_face, coord], axis=0)
        else:
            coord = np.concatenate([coord_body, coord], axis=0)

        ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
        ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)
        rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)

        pts3d = pts3d[coord[:, 0], coord[:, 1]]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d = np.linalg.norm(pts3d - ray_o, axis=1) / norm_ray
        near = d.astype(np.float32)
        far = near + 0.5

        mask_at_box = np.ones_like(near)
    else:
        coord = np.argwhere(dpt_msk == 1)

        rgb = img[coord[:, 0], coord[:, 1]].astype(np.float32)
        ray_o = ray_o[coord[:, 0], coord[:, 1]].astype(np.float32)
        ray_d = ray_d[coord[:, 0], coord[:, 1]].astype(np.float32)

        pts3d = pts3d[coord[:, 0], coord[:, 1]]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d = np.linalg.norm(pts3d - ray_o, axis=1) / norm_ray
        near = (d - 0.25).astype(np.float32)
        far = (d + 0.25).astype(np.float32)

        mask_at_box = (dpt_msk == 1).ravel()

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    transformed_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - transformed_joints
    transforms = transforms.astype(np.float32)

    return transforms
