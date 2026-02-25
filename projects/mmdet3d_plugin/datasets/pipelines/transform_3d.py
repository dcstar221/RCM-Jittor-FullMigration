
import numpy as np
import jittor as jt
from PIL import Image
from projects.mmdet3d_plugin.jittor_adapter import PIPELINES


@PIPELINES.register_module()
class GlobalRotScaleTrans:
    def __init__(self, rot_range, scale_ratio_range, translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        noise_scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        noise_translate = np.array([
            np.random.normal(0, self.translation_std[0]),
            np.random.normal(0, self.translation_std[1]),
            np.random.normal(0, self.translation_std[2])
        ])
        if 'points' in results and results['points'].shape[0] > 0:
            points = results['points']
            rot_sin = np.sin(noise_rotation)
            rot_cos = np.cos(noise_rotation)
            rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            points[:, :3] = points[:, :3] @ rot_mat.T
            points[:, :3] *= noise_scale
            points[:, :3] += noise_translate
            results['points'] = points
        if 'gt_bboxes_3d' in results and results['gt_bboxes_3d'].shape[0] > 0:
            bboxes = results['gt_bboxes_3d']
            rot_sin = np.sin(noise_rotation)
            rot_cos = np.cos(noise_rotation)
            rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            bboxes[:, :3] = bboxes[:, :3] @ rot_mat.T
            bboxes[:, 6] += noise_rotation
            bboxes[:, :6] *= noise_scale
            bboxes[:, :3] += noise_translate
            results['gt_bboxes_3d'] = bboxes
        return results


@PIPELINES.register_module()
class RandomFlip3D:
    def __init__(self, flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=0.0):
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def __call__(self, results):
        if np.random.rand() < self.flip_ratio_bev_horizontal:
            if 'points' in results and results['points'].shape[0] > 0:
                results['points'][:, 1] = -results['points'][:, 1]
            if 'gt_bboxes_3d' in results and results['gt_bboxes_3d'].shape[0] > 0:
                results['gt_bboxes_3d'][:, 1] = -results['gt_bboxes_3d'][:, 1]
                results['gt_bboxes_3d'][:, 6] = -results['gt_bboxes_3d'][:, 6]
        return results


@PIPELINES.register_module()
class PointsRangeFilter:
    def __init__(self, point_cloud_range):
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, results):
        if 'points' not in results or results['points'].shape[0] == 0:
            return results
        points = results['points']
        mask = (points[:, 0] >= self.point_cloud_range[0]) & \
               (points[:, 1] >= self.point_cloud_range[1]) & \
               (points[:, 2] >= self.point_cloud_range[2]) & \
               (points[:, 0] <= self.point_cloud_range[3]) & \
               (points[:, 1] <= self.point_cloud_range[4]) & \
               (points[:, 2] <= self.point_cloud_range[5])
        results['points'] = points[mask]
        return results


@PIPELINES.register_module()
class ObjectRangeFilter:
    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, results):
        if 'gt_bboxes_3d' in results and results['gt_bboxes_3d'].shape[0] > 0:
            gt_bboxes_3d = results['gt_bboxes_3d']
            gt_labels_3d = results['gt_labels_3d']
            mask = (gt_bboxes_3d[:, 0] >= self.pcd_range[0]) & \
                   (gt_bboxes_3d[:, 1] >= self.pcd_range[1]) & \
                   (gt_bboxes_3d[:, 0] <= self.pcd_range[3]) & \
                   (gt_bboxes_3d[:, 1] <= self.pcd_range[4])
            results['gt_bboxes_3d'] = gt_bboxes_3d[mask]
            results['gt_labels_3d'] = gt_labels_3d[mask]
        return results


@PIPELINES.register_module()
class ObjectNameFilter:
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class PointShuffle:
    def __call__(self, results):
        if 'points' in results and results['points'].shape[0] > 0:
            np.random.shuffle(results['points'])
        return results


@PIPELINES.register_module()
class DefaultFormatBundle3D:
    def __init__(self, class_names, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        from projects.mmdet3d_plugin.core.bbox.structures import LiDARInstance3DBoxes
        
        # Add box_type_3d to results so Collect3D can pick it up or it persists
        # Usually it goes into img_metas
        if 'img_metas' not in results:
            results['img_metas'] = {}
        results['img_metas']['box_type_3d'] = LiDARInstance3DBoxes
        results['box_type_3d'] = LiDARInstance3DBoxes # Also put in top level just in case

        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = np.array(results['gt_labels_3d']).astype(np.int64)
        if 'img' in results:
            imgs = results['img']
            if isinstance(imgs, list) and len(imgs) > 0:
                # Stack and transpose to (N, 3, H, W)
                # Images might have different shapes after augmentations
                # For safety, use the shape of the first image
                h, w = imgs[0].shape[:2]
                stacked = []
                for img in imgs:
                    if img.shape[:2] != (h, w):
                        pil_img = Image.fromarray(img.astype(np.uint8))
                        pil_img = pil_img.resize((w, h), Image.BILINEAR)
                        img = np.array(pil_img).astype(np.float32)
                    stacked.append(img)
                stacked = np.stack(stacked, axis=0)  # (N, H, W, 3)
                stacked = stacked.transpose(0, 3, 1, 2)  # (N, 3, H, W)
                results['img'] = stacked.astype(np.float32)
        return results


@PIPELINES.register_module()
class NormalizeMultiviewImage:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        imgs = results.get('img', [])
        new_imgs = []
        for img in imgs:
            img = img.astype(np.float32)
            img = (img - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
            new_imgs.append(img)
        results['img'] = new_imgs
        return results


@PIPELINES.register_module()
class PadMultiViewImage:
    def __init__(self, size_divisor=None, size=None):
        self.size_divisor = size_divisor
        self.size = size

    def __call__(self, results):
        imgs = results.get('img', [])
        new_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            if self.size_divisor:
                pad_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor - h
                pad_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor - w
            else:
                pad_h = max(self.size[0] - h, 0)
                pad_w = max(self.size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            new_imgs.append(img)
        results['img'] = new_imgs
        results['pad_shape'] = [img.shape for img in new_imgs]
        return results


@PIPELINES.register_module()
class RandomScaleImageMultiViewImageCus:
    def __init__(self, scales=[0.5, 0.5]):
        self.scales = scales
        
    def __call__(self, results):
        scale = np.random.choice(self.scales)
        imgs = results.get('img', [])
        new_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            new_img = np.array(pil_img).astype(np.float32)
            new_imgs.append(new_img)
        results['img'] = new_imgs
        if 'cam_intrinsic' in results:
            new_intrinsics = []
            for intr in results['cam_intrinsic']:
                intr_new = intr.copy()
                intr_new[:2] *= scale
                new_intrinsics.append(intr_new)
            results['cam_intrinsic'] = new_intrinsics
        return results


# --- Stub transforms referenced in config but not yet fully implemented ---

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Photo metric distortion for multi-view images (stub: no-op for now)."""
    def __init__(self, **kwargs):
        pass
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class MultCamImageAugmentation:
    """Multi-camera image augmentation (stub: no-op for now)."""
    def __init__(self, ida_aug_conf=None, **kwargs):
        self.ida_aug_conf = ida_aug_conf
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class MultiModalBEVAugmentation:
    """Multi-modal BEV augmentation (stub: no-op for now)."""
    def __init__(self, bda_aug_conf=None, **kwargs):
        self.bda_aug_conf = bda_aug_conf
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class CustomCollect3D:
    """Collect data from the loader for a 3D task."""
    def __init__(self, keys, meta_keys=None):
        self.keys = keys
        self.meta_keys = meta_keys or []

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        # Collect meta info
        meta = {}
        for key in self.meta_keys:
            if key in results:
                meta[key] = results[key]
        if meta:
            data['img_metas'] = meta
        # Also pass through useful meta even if not requested
        for pass_key in ['img_shape', 'ori_shape', 'pad_shape', 'lidar2img', 
                         'cam_intrinsic', 'img_filename', 'filename']:
            if pass_key in results:
                data.setdefault('img_metas', {})
                data['img_metas'][pass_key] = results[pass_key]
        return data


@PIPELINES.register_module()
class MultiScaleFlipAug3D:
    """Multi-scale flip augmentation for 3D (simplified: applies inner transforms)."""
    def __init__(self, img_scale=None, pts_scale_ratio=1, flip=False, transforms=None, **kwargs):
        self.img_scale = img_scale
        self.pts_scale_ratio = pts_scale_ratio
        self.flip = flip
        self.transforms = []
        if transforms:
            from projects.mmdet3d_plugin.jittor_adapter import build_from_cfg
            for t in transforms:
                if isinstance(t, dict):
                    self.transforms.append(build_from_cfg(t, PIPELINES))
                elif callable(t):
                    self.transforms.append(t)

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results
