
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
# from mmcv.parallel import DataContainer as DC
import jittor as jt

# from mmdet3d.core.bbox import BaseInstance3DBoxes
# from mmdet3d.core.points import BasePoints
from projects.mmdet3d_plugin.jittor_adapter import PIPELINES
# from mmdet.datasets.pipelines import to_tensor
# from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

def to_tensor(data):
    """Convert objects of various python types to :obj:`jittor.Var`."""
    if isinstance(data, jt.Var):
        return data
    elif isinstance(data, np.ndarray):
        return jt.array(data)
    elif isinstance(data, int):
        return jt.array(data)
    elif isinstance(data, float):
        return jt.array(data)
    elif isinstance(data, list):
        # Check if list of arrays or list of numbers
        if len(data) > 0 and isinstance(data[0], (int, float)):
            return jt.array(data)
        elif len(data) > 0 and isinstance(data[0], np.ndarray):
            return [jt.array(x) for x in data]
        else:
            return data
    else:
        return data

@PIPELINES.register_module()
class DefaultFormatBundle3D:
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor
    - proposals: (1)to tensor
    - gt_bboxes: (1)to tensor
    - gt_bboxes_ignore: (1)to tensor
    - gt_labels: (1)to tensor
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'img' in results:
            if isinstance(results['img'], list):
                # Multiple images
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                results['img'] = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results['img'] = np.ascontiguousarray(img.transpose(2, 0, 1))
        
        if 'gt_bboxes_3d' in results:
             # Already handled by specific loaders or just kept as object
             pass

        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = to_tensor(results['gt_labels_3d'])

        return results

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor
    - proposals: (1)to tensor
    - gt_bboxes: (1)to tensor
    - gt_bboxes_ignore: (1)to tensor
    - gt_labels: (1)to tensor
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        if 'gt_map_masks' in results:
            results['gt_map_masks'] = to_tensor(results['gt_map_masks'])

        return results