# Copyright (c) OpenMMLab. All rights reserved.
import jittor as jt
import jittor.nn as nn
import warnings
from projects.mmdet3d_plugin.jittor_adapter import (
    DETECTORS, BaseModule, build_backbone, build_neck, 
    build_head, build_voxel_encoder, build_middle_encoder, 
    build_fusion_layer
)
# We assume these builders are available or we need to implement them. 
# jittor_adapter.py has registries but maybe not build functions exposed nicely?
# Let's check jittor_adapter.py content again. It has build_from_cfg.

from projects.mmdet3d_plugin.jittor_adapter import build_from_cfg, BACKBONES, NECKS, HEADS, VOXEL_ENCODERS, MIDDLE_ENCODERS, FUSION_LAYERS

def build_backbone(cfg): return build_from_cfg(cfg, BACKBONES)
def build_neck(cfg): return build_from_cfg(cfg, NECKS)
def build_head(cfg): return build_from_cfg(cfg, HEADS)
def build_voxel_encoder(cfg): return build_from_cfg(cfg, VOXEL_ENCODERS)
def build_middle_encoder(cfg): return build_from_cfg(cfg, MIDDLE_ENCODERS)
def build_fusion_layer(cfg): return build_from_cfg(cfg, FUSION_LAYERS)

class Base3DDetector(BaseModule):
    """Base class for detectors."""
    def __init__(self, init_cfg=None):
        super(Base3DDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass

@DETECTORS.register_module()
class MVXTwoStageDetectorCustomJittor(Base3DDetector):
    """Base class of Multi-modality VoxelNet (Jittor Version)."""

    def __init__(self,
                 freeze_img=False,
                 freeze_pts=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MVXTwoStageDetectorCustomJittor, self).__init__(init_cfg=init_cfg)
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        
        # pts_voxel_layer is ignored in Jittor version as voxelization is done in encoder
        self.pts_voxel_layer = None
        self.pts_fusion_layer = None
        
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_fusion_layer:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            # Update cfg logic might need adjustment if pts_fusion_layer is a dict
            if isinstance(pts_fusion_layer, dict):
                pts_fusion_layer['train_cfg'] = pts_train_cfg
                pts_test_cfg = test_cfg.pts if test_cfg else None
                pts_fusion_layer['test_cfg'] = pts_test_cfg
            self.pts_fusion_layer = build_fusion_layer(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            if isinstance(pts_bbox_head, dict):
                pts_bbox_head['train_cfg'] = pts_train_cfg
                pts_test_cfg = test_cfg.pts if test_cfg else None
                pts_bbox_head['test_cfg'] = pts_test_cfg
            self.pts_bbox_head = build_head(pts_bbox_head)
        
        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        if img_rpn_head is not None:
            # skipped for now
            pass
        if img_roi_head is not None:
            self.img_roi_head = build_head(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Freezing logic
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
        if self.freeze_pts:
            if self.with_voxel_encoder:
                for param in self.pts_voxel_encoder.parameters():
                    param.requires_grad = False
            if self.with_middle_encoder:
                for param in self.pts_middle_encoder.parameters():
                    param.requires_grad = False
            if self.with_pts_backbone:
                for param in self.pts_backbone.parameters():
                    param.requires_grad = False
            if self.with_pts_neck:
                for param in self.pts_neck.parameters():
                    param.requires_grad = False
            if self.with_pts_bbox:
                for param in self.pts_bbox_head.parameters():
                    param.requires_grad = False

    @property
    def with_pts_bbox(self):
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        return hasattr(self, 'pts_fusion_layer') and self.pts_fusion_layer is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_voxel_encoder(self):
        return hasattr(self, 'pts_voxel_encoder') and self.pts_voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        return hasattr(self, 'pts_middle_encoder') and self.pts_middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            # Jittor image shape: [N, C, H, W] or [B, N, C, H, W]
            if img.ndim == 5 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.ndim == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.view(B * N, C, H, W)
            
            img_feats = self.img_backbone(img)
            # img_feats is usually a tuple/list of features
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        
        # Jittor DynamicPillarVFESimple2D handles voxelization
        # Input pts is list of [N, C] arrays/tensors
        voxel_features, coors = self.pts_voxel_encoder(pts)
        
        # coors: [batch_idx, z, y, x]
        # batch_size needed for middle encoder
        if coors.shape[0] > 0:
            batch_size = int(coors[:, 0].max().item()) + 1
        else:
            batch_size = 1 # Default or handle empty
            
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, 
                      gt_labels_3d=None, gt_labels=None, gt_bboxes=None, 
                      img=None, proposals=None, gt_bboxes_ignore=None):
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
