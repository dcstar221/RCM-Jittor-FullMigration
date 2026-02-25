
import jittor as jt
import jittor.nn as nn
import numpy as np
import copy
from projects.mmdet3d_plugin.jittor_adapter import DETECTORS
from .mvx_two_stage_custom_jittor import MVXTwoStageDetectorCustomJittor
# from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask # Pending Jittor port

@DETECTORS.register_module()
class RCMFusionJittor(MVXTwoStageDetectorCustomJittor):
    """RCMFusion for Jittor.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
        video_train_mode (bool): Decide whether to use temporal information during train.
    """

    def __init__(self,
                 freeze_img=False,
                 freeze_pts=False,
                 use_grid_mask=False,
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
                 video_test_mode=False,
                 video_train_mode=False,
                 init_cfg=None
                 ):
        super(RCMFusionJittor,
              self).__init__(freeze_img=freeze_img, 
                             freeze_pts=freeze_pts, 
                             pts_voxel_layer=pts_voxel_layer, 
                             pts_voxel_encoder=pts_voxel_encoder,
                             pts_middle_encoder=pts_middle_encoder, 
                             pts_fusion_layer=pts_fusion_layer,
                             img_backbone=img_backbone, 
                             pts_backbone=pts_backbone, 
                             img_neck=img_neck, 
                             pts_neck=pts_neck,
                             pts_bbox_head=pts_bbox_head, 
                             img_roi_head=img_roi_head, 
                             img_rpn_head=img_rpn_head,
                             train_cfg=train_cfg, 
                             test_cfg=test_cfg, 
                             pretrained=pretrained,
                             init_cfg=init_cfg)
        
        # GridMask pending implementation
        # self.grid_mask = GridMask(
        #     True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.video_train_mode = video_train_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract point features."""
        print("DEBUG: extract_pts_feat start")
        if not self.with_pts_bbox:
            return None
        
        # 1. Voxelize
        print("DEBUG: Calling pts_voxel_encoder")
        voxel_features, pillar_coors = self.pts_voxel_encoder(pts)
        print(f"DEBUG: Voxel features shape: {voxel_features.shape}, coors: {pillar_coors.shape}")
        
        # 2. Backbone (Scatter + ResNet)
        batch_size = len(pts)
        print("DEBUG: Calling pts_backbone")
        x = self.pts_backbone(voxel_features, pillar_coors, batch_size)
        print("DEBUG: pts_backbone done")
        
        # 3. Neck (FPN)
        if self.with_pts_neck:
            print("DEBUG: Calling pts_neck")
            x = self.pts_neck(x, batch_size)
            print("DEBUG: pts_neck done")
            
        return x

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.shape[0]
        print(f"DEBUG: extract_img_feat input img shape: {img.shape}, B={B}")
        if img is not None:
            # Jittor dim check
            if img.ndim == 5 and img.shape[0] == 1:
                img = img.squeeze(0)
                print(f"DEBUG: img squeezed to {img.shape}")
            elif img.ndim == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.reshape(B * N, C, H, W)
                print(f"DEBUG: img reshaped to {img.shape}")
            
            # if self.use_grid_mask:
            #     img = self.grid_mask(img)
            
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
            
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            print(f"DEBUG: img_feat shape before reshape: {img_feat.shape}, B={B}, BN={BN}")
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            print(f"DEBUG: img_feat reshaped to {img_feats_reshaped[-1].shape}")
        return img_feats_reshaped

    def extract_feat(self, img, points, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        pts_feats = self.extract_pts_feat(points)
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats, pts_feats

    def execute(self, return_loss=True, **kwargs):
        """Calls either forward_train or simple_test depending on return_loss."""
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.simple_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function."""
        # Placeholder for training logic
        pass

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentation."""
        img_feats, pts_feats = self.extract_feat(img, points, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if self.with_pts_bbox:
            bbox_outputs = self.pts_bbox_head(img_feats, img_metas, pts_bev=pts_feats)
            # For verification, just return the raw outputs
            return bbox_outputs
            
        return bbox_list


    def get_corners_from_tensor(self, boxes):
        """
        boxes: [N, 7] (x, y, z, dx, dy, dz, heading)
        Return: [N, 8, 3]
        """
        if boxes.shape[0] == 0:
            return jt.zeros((0, 8, 3))
            
        dx = boxes[:, 3]
        dy = boxes[:, 4]
        dz = boxes[:, 5]
        heading = boxes[:, 6]
        
        dx2 = dx / 2
        dy2 = dy / 2
        dz2 = dz / 2
        
        # 0: (+dx/2, +dy/2, -dz/2)
        # 1: (+dx/2, -dy/2, -dz/2)
        # 2: (-dx/2, -dy/2, -dz/2)
        # 3: (-dx/2, +dy/2, -dz/2)
        # 4: (+dx/2, +dy/2, +dz/2)
        # 5: (+dx/2, -dy/2, +dz/2)
        # 6: (-dx/2, -dy/2, +dz/2)
        # 7: (-dx/2, +dy/2, +dz/2)
        
        x_corners = jt.stack([
             dx2,  dx2, -dx2, -dx2,
             dx2,  dx2, -dx2, -dx2
        ], dim=1)
        
        y_corners = jt.stack([
             dy2, -dy2, -dy2,  dy2,
             dy2, -dy2, -dy2,  dy2
        ], dim=1)
        
        z_corners = jt.stack([
            -dz2, -dz2, -dz2, -dz2,
             dz2,  dz2,  dz2,  dz2
        ], dim=1)
        
        c = jt.cos(heading).unsqueeze(1)
        s = jt.sin(heading).unsqueeze(1)
        
        x_new = x_corners * c - y_corners * s
        y_new = x_corners * s + y_corners * c
        z_new = z_corners
        
        x_new = x_new + boxes[:, 0].unsqueeze(1)
        y_new = y_new + boxes[:, 1].unsqueeze(1)
        z_new = z_new + boxes[:, 2].unsqueeze(1)
        
        corners = jt.stack([x_new, y_new, z_new], dim=2)
        return corners

    def make_polar_coordinate(self, points, bbox_list):
        # bbox_list structure depends on head implementation.
        # In RCMFusion.py: bbox_list[i][0] is the result (bboxes, scores, labels)?
        # No, get_bboxes usually returns list of tuples (bboxes, scores, labels).
        # RCMFusion.py line 132: box_num = bbox_list[0][0].tensor.shape[0]
        # This implies bbox_list[0] is (bboxes, ...). bboxes has .tensor
        # Jittor BBox3D might differ. We need to check what get_bboxes returns in Jittor adapter.
        
        # Assuming bbox_list is list of (bboxes, scores, labels)
        # And bboxes has .tensor property or is a tensor.
        
        if len(bbox_list) == 0:
             return [], []
             
        # Use first sample to check structure
        first_box = bbox_list[0][0] # This should be the boxes object/tensor
        if hasattr(first_box, 'tensor'):
            box_num = first_box.tensor.shape[0]
        else:
            box_num = first_box.shape[0] # If it's just tensor
            
        bs = len(points)
        # mask = [True, False, False, True, True, False, False, True] # For corners?
        # RCMFusion logic:
        # cartesian_corners = bbox_list[i][0].corners[:,mask][...,:2]
        
        batch_polar_points = []
        batch_polar_corners = [] 
        
        for i in range(bs):
            # Points: [N, C]
            # Polar conversion for points
            # r = sqrt(x^2 + y^2)
            # theta = atan2(y, x)
            pts_x = points[i][:, 0]
            pts_y = points[i][:, 1]
            pts_range = jt.sqrt(pts_x**2 + pts_y**2)
            pts_angle = jt.atan2(pts_y, pts_x)
            
            tmp_point = points[i][:, :6].clone()
            tmp_point[:, 0] = pts_range
            tmp_point[:, 1] = pts_angle
            batch_polar_points.append(tmp_point)

            # Box corners polar conversion
            # We need corners of the boxes.
            # If bbox_list[i][0] is LiDARInstance3DBoxes, it has .corners property -> [N, 8, 3]
            bboxes = bbox_list[i][0]
            if hasattr(bboxes, 'corners'):
                corners = bboxes.corners # [N, 8, 3]
            else:
                corners = self.get_corners_from_tensor(bboxes)
                
            # mask = [True, False, False, True, True, False, False, True] (indices 0,3,4,7)
            # These are usually the bottom 4 corners for LiDAR boxes?
            # 0: front-left-bottom, 1: front-left-top...
            # We need to verify corner order. 
            # OpenPCDet/MMDet3D: 0:FLB, 1:FLT, 2:BLT, 3:BLB... 
            # RCMFusion mask selects 4 corners. 
            # Let's trust RCMFusion indices: 0, 3, 4, 7.
            
            mask_indices = [0, 3, 4, 7]
            if isinstance(corners, jt.Var):
                cartesian_corners = corners[:, mask_indices][..., :2] # [N, 4, 2]
            else:
                cartesian_corners = jt.array(corners[:, mask_indices][..., :2])
            
            # cartesian_corners: [N, 4, 2] (x, y)
            
            # range_max, range_min, angle_max, angle_min
            dists = jt.sqrt(jt.sum(cartesian_corners**2, dim=2)) # [N, 4]
            range_max = jt.max(dists, dim=1, keepdims=True) # [N, 1]
            range_min = jt.min(dists, dim=1, keepdims=True) # [N, 1]
            
            angles = jt.atan2(cartesian_corners[..., 1], cartesian_corners[..., 0]) # [N, 4]
            # Wrap angles logic from RCMFusion:
            # angle_max = max(angle + pi) - pi
            # This logic seems to handle -pi/pi wrap around?
            
            angles_shifted = angles + np.pi
            angle_max = jt.max(angles_shifted, dim=1, keepdims=True) - np.pi
            angle_min = jt.min(angles_shifted, dim=1, keepdims=True) - np.pi
            
            polar_corner = jt.concat([range_max, range_min, angle_max, angle_min], dim=1) # [N, 4]
            batch_polar_corners.append(polar_corner)
            
        # Stack corners
        # RCMFusion does cat then view.
        # batch_polar_corners = torch.cat(..., dim=0) -> (bs*box_num, 4) -> view(bs, box_num, 4)
        if len(batch_polar_corners) > 0:
            batch_polar_corners = jt.stack(batch_polar_corners, dim=0) # [bs, box_num, 4]
        else:
            batch_polar_corners = jt.zeros((0, 4)) # Handle empty?

        return batch_polar_points, batch_polar_corners

    def forward_pts_train(self,
                          img_feats,
                          points,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          pts_feats=None):
        """Forward function"""
        # pts_bbox_head(img_feats, img_metas, prev_bev, pts_feats)
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev, pts_feats)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
        
        polar_points, box_corner_polar = self.make_polar_coordinate(points, bbox_list)
        
        refine_point = {'original' : points, 'polar' : polar_points}
        refine_box = {'boxes' : bbox_list, 'polar_corner' :  box_corner_polar}
        
        if self.pts_fusion_layer is not None:
            outs = self.pts_fusion_layer(outs, refine_point, refine_box, img_feats, img_metas)
            
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, losses]
            final_losses = self.pts_fusion_layer.loss(*loss_inputs, img_metas=img_metas)
            return final_losses
        else:
            return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        # Clean cache if needed? torch.cuda.empty_cache()
        jt.gc()
        
        len_queue = img.shape[1]
        if self.video_train_mode:
            # prev_img = img[:, :-1, ...]
            # Not implementing temporal train for now unless requested
            prev_bev = None 
            # Implement obtain_history_bev if needed
        else:
            prev_bev = None
            
        img = img[:, -1, ...] # (B, N, C, H, W)
        # points = points[-1] # if points has temporal dim? 
        # Usually points is list of tensors, one per sample.
        # If queue, points might be list of lists?
        # RCMFusion.py line 281 commented out: # points = points[-1]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        
        img_feats, pts_feats = self.extract_feat(img, points, img_metas=img_metas)
        
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, points, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, pts_feats)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        # Basic test forward
        if not isinstance(img_metas, list):
             img_metas = [img_metas]
        
        # Ensure img is a list if it's not None
        if img is not None and not isinstance(img, list):
            img = [img]
        elif img is None:
            img = [None]
        
        # Temporal logic simplified for initial test
        # Just use simple_test
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, points=None):
        # RCMFusion simple_test
        _points = []
        if points is not None:
             # points is [[p1, p2...]]? 
             # points arg in simple_test comes from **kwargs?
             # Usually simple_test(img, img_metas, points=...)
             _points.append(points[0]) # Assuming batch size 1 for test?
        
        img_feats, pts_feats = self.extract_feat(img=img, points=_points, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        
        # simple_test_pts
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, _points, img_metas, prev_bev, pts_feats, rescale=rescale)
            
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return new_prev_bev, bbox_list

    def simple_test_pts(self, img_feats, points, img_metas, prev_bev=None, pts_feats=None, rescale=False):
        """Test function of point cloud branch."""
        # pts_bbox_head expects img_metas as a list of dicts (batch mode)
        # In simple_test, img_metas is a single dict, so we wrap it
        img_metas_list = [img_metas] if not isinstance(img_metas, list) else img_metas
        
        outs = self.pts_bbox_head(img_feats, img_metas_list, prev_bev, pts_feats)
        
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas_list, rescale=rescale)
        
        polar_points, box_corner_polar = self.make_polar_coordinate(points, bbox_list)
        
        refine_point = {'original' : points, 'polar' : polar_points}
        refine_box = {'boxes' : bbox_list, 'polar_corner' :  box_corner_polar}
        
        if self.pts_fusion_layer is not None:
            outs = self.pts_fusion_layer(outs, refine_point, refine_box, img_feats, img_metas_list)
            # bbox_list_ = self.pts_fusion_layer.get_bboxes(outs, img_metas_list, rescale=rescale)
            
            # Manually update bbox_list with refined boxes from InstanceLevelFusion
            # outs['all_bbox_preds_refine'] shape: (1, bs, num_box, dim)
            refined_boxes_batch = outs['all_bbox_preds_refine'][0] 
            
            new_bbox_list = []
            for i, res_tuple in enumerate(bbox_list):
                # Unpack based on length (FeatureLevelFusion returns [bboxes, head_feats, scores, labels] or [bboxes, scores, labels])
                if len(res_tuple) == 4:
                    bboxes, head_feats, scores, labels = res_tuple
                elif len(res_tuple) == 3:
                    bboxes, scores, labels = res_tuple
                else:
                    # Fallback or unknown format
                    bboxes = res_tuple[0]
                    scores = res_tuple[-2]
                    labels = res_tuple[-1]
                
                # Get refined boxes for this batch
                refined = refined_boxes_batch[i]
                
                # Ensure refined boxes match original boxes count
                if refined.shape[0] != bboxes.shape[0]:
                    print(f"WARNING: Refined boxes count {refined.shape[0]} != original {bboxes.shape[0]}")
                    # If mismatch, maybe just use original? Or slice?
                    # Should match if logic is correct.
                
                new_bbox_list.append((refined, scores, labels))
            
            bbox_list_ = new_bbox_list
        else:
            # If no fusion layer, use the bbox_list from pts_bbox_head
            # But we need to ensure bbox_list_ format is (bboxes, scores, labels)
            # bbox_list might contain head_features
            new_bbox_list = []
            for res_tuple in bbox_list:
                if len(res_tuple) == 4:
                    bboxes, head_feats, scores, labels = res_tuple
                    new_bbox_list.append((bboxes, scores, labels))
                else:
                    new_bbox_list.append(res_tuple)
            bbox_list_ = new_bbox_list
        
        # bbox3d2result
        bbox_results = [
            (bboxes, scores, labels) # bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list_
        ]
        return outs['bev_embed'], bbox_results
