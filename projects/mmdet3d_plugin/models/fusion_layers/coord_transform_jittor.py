import jittor as jt
import numpy as np
from functools import partial

def apply_3d_transformation(pcd, coord_type, img_meta, reverse=False):
    """Apply transformation to input point cloud (Jittor version).

    Args:
        pcd (jt.Var): The point cloud to be transformed. Shape (N, 3+C).
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Returns:
        jt.Var: The transformed point cloud.
    """
    # Ensure pcd is a Jittor array
    if not isinstance(pcd, jt.Var):
        pcd = jt.array(pcd)

    dtype = pcd.dtype
    
    # Extract transformation parameters from img_meta
    if 'pcd_rotation' in img_meta:
        pcd_rotate_mat = jt.array(img_meta['pcd_rotation']).astype(dtype)
    else:
        pcd_rotate_mat = jt.eye(3, dtype=dtype)

    if 'pcd_scale_factor' in img_meta:
        pcd_scale_factor = float(img_meta['pcd_scale_factor'])
    else:
        pcd_scale_factor = 1.0

    if 'pcd_trans' in img_meta:
        pcd_trans_factor = jt.array(img_meta['pcd_trans']).astype(dtype)
    else:
        pcd_trans_factor = jt.zeros((3,), dtype=dtype)

    pcd_horizontal_flip = img_meta.get('pcd_horizontal_flip', False)
    pcd_vertical_flip = img_meta.get('pcd_vertical_flip', False)

    flow = img_meta.get('transformation_3d_flow', [])

    pcd = pcd.clone()
    
    # Helper functions for transformations
    def scale_func(points, scale):
        points[:, :3] *= scale
        return points

    def translate_func(points, trans):
        points[:, :3] += trans
        return points

    def rotate_func(points, rot_mat):
        # points: (N, 3+C), rot_mat: (3, 3)
        # points[:, :3] = points[:, :3] @ rot_mat.transpose()
        points[:, :3] = jt.matmul(points[:, :3], rot_mat.transpose())
        return points

    def flip_func(points, direction):
        # Assumes LIDAR coordinates (x, y, z)
        # horizontal flip: flip along y-axis (y = -y)
        # vertical flip: flip along x-axis (x = -x)
        if direction == 'horizontal':
            points[:, 1] = -points[:, 1]
        elif direction == 'vertical':
            points[:, 0] = -points[:, 0]
        return points

    # Define operations
    if reverse:
        ops = []
        # Reverse order and invert operations
        for op in reversed(flow):
            if op == 'S':
                ops.append(partial(scale_func, scale=1.0 / pcd_scale_factor))
            elif op == 'T':
                ops.append(partial(translate_func, trans=-pcd_trans_factor))
            elif op == 'R':
                # Inverse rotation is transpose for rotation matrix
                ops.append(partial(rotate_func, rot_mat=pcd_rotate_mat.transpose()))
            elif op == 'HF':
                if pcd_horizontal_flip:
                    ops.append(partial(flip_func, direction='horizontal'))
            elif op == 'VF':
                if pcd_vertical_flip:
                    ops.append(partial(flip_func, direction='vertical'))
    else:
        ops = []
        for op in flow:
            if op == 'S':
                ops.append(partial(scale_func, scale=pcd_scale_factor))
            elif op == 'T':
                ops.append(partial(translate_func, trans=pcd_trans_factor))
            elif op == 'R':
                ops.append(partial(rotate_func, rot_mat=pcd_rotate_mat))
            elif op == 'HF':
                if pcd_horizontal_flip:
                    ops.append(partial(flip_func, direction='horizontal'))
            elif op == 'VF':
                if pcd_vertical_flip:
                    ops.append(partial(flip_func, direction='vertical'))

    # Apply operations
    for func in ops:
        pcd = func(pcd)

    return pcd


def extract_2d_info(img_meta, tensor):
    """Extract image augmentation information from img_meta.
    """
    img_shape = img_meta['img_shape']
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = img_shape
    ori_h, ori_w, _ = ori_shape

    if 'scale_factor' in img_meta:
        scale_factor = img_meta['scale_factor']
        if isinstance(scale_factor, np.ndarray):
            scale_factor = scale_factor[:2]
        elif isinstance(scale_factor, list):
            scale_factor = scale_factor[:2]
        else:
             scale_factor = [scale_factor, scale_factor]
        img_scale_factor = jt.array(scale_factor)
    else:
        img_scale_factor = jt.array([1.0, 1.0])

    img_flip = img_meta.get('flip', False)
    
    if 'img_crop_offset' in img_meta:
        img_crop_offset = jt.array(img_meta['img_crop_offset'])
    else:
        img_crop_offset = jt.array([0.0, 0.0])

    return (img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip, img_crop_offset)


def bbox_2d_transform(img_meta, bbox_2d, ori2new):
    """Transform 2d bbox according to img_meta (Jittor version).
    """
    if not isinstance(bbox_2d, jt.Var):
        bbox_2d = jt.array(bbox_2d)

    img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip, \
        img_crop_offset = extract_2d_info(img_meta, bbox_2d)

    bbox_2d_new = bbox_2d.clone()

    if ori2new:
        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] * img_scale_factor[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] * img_scale_factor[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] * img_scale_factor[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] * img_scale_factor[1]

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] + img_crop_offset[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] + img_crop_offset[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] + img_crop_offset[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] + img_crop_offset[1]

        if img_flip:
            bbox_2d_r = img_w - bbox_2d_new[:, 0]
            bbox_2d_l = img_w - bbox_2d_new[:, 2]
            bbox_2d_new[:, 0] = bbox_2d_l
            bbox_2d_new[:, 2] = bbox_2d_r
    else:
        if img_flip:
            bbox_2d_r = img_w - bbox_2d_new[:, 0]
            bbox_2d_l = img_w - bbox_2d_new[:, 2]
            bbox_2d_new[:, 0] = bbox_2d_l
            bbox_2d_new[:, 2] = bbox_2d_r

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] - img_crop_offset[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] - img_crop_offset[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] - img_crop_offset[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] - img_crop_offset[1]

        bbox_2d_new[:, 0] = bbox_2d_new[:, 0] / img_scale_factor[0]
        bbox_2d_new[:, 2] = bbox_2d_new[:, 2] / img_scale_factor[0]
        bbox_2d_new[:, 1] = bbox_2d_new[:, 1] / img_scale_factor[1]
        bbox_2d_new[:, 3] = bbox_2d_new[:, 3] / img_scale_factor[1]

    return bbox_2d_new


def coord_2d_transform(img_meta, coord_2d, ori2new):
    """Transform 2d pixel coordinates according to img_meta (Jittor version).
    """
    if not isinstance(coord_2d, jt.Var):
        coord_2d = jt.array(coord_2d)

    img_h, img_w, ori_h, ori_w, img_scale_factor, img_flip, \
        img_crop_offset = extract_2d_info(img_meta, coord_2d)

    coord_2d_new = coord_2d.clone()

    if ori2new:
        coord_2d_new[..., 0] = coord_2d_new[..., 0] * img_scale_factor[0]
        coord_2d_new[..., 1] = coord_2d_new[..., 1] * img_scale_factor[1]

        coord_2d_new[..., 0] += img_crop_offset[0]
        coord_2d_new[..., 1] += img_crop_offset[1]

        # flip uv coordinates
        if img_flip:
            coord_2d_new[..., 0] = img_w - coord_2d_new[..., 0]
    else:
        if img_flip:
            coord_2d_new[..., 0] = img_w - coord_2d_new[..., 0]

        coord_2d_new[..., 0] -= img_crop_offset[0]
        coord_2d_new[..., 1] -= img_crop_offset[1]

        coord_2d_new[..., 0] = coord_2d_new[..., 0] / img_scale_factor[0]
        coord_2d_new[..., 1] = coord_2d_new[..., 1] / img_scale_factor[1]

    return coord_2d_new
