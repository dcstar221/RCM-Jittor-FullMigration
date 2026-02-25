import jittor as jt
from jittor import nn
import numpy as np
from projects.mmdet3d_plugin.jittor_adapter import FUSION_LAYERS
from projects.mmdet3d_plugin.ops.pointnet_utils_jittor import build_sa_module, furthest_point_sample
from projects.mmdet3d_plugin.rcm_fusion.modules.detr3d_cross_attention_jittor import Detr3DCrossAtten
import copy
import time

def SoftPolarAssociation(polar_pts, polar_corners_):
    # proposal-wise grouping range
    point_num = int(polar_pts.shape[0])
    num_box = int(polar_corners_.shape[0])
    range_center = (polar_corners_[:,0] + polar_corners_[:,1])/2
    
    box_length = jt.abs(polar_corners_[:,0] - polar_corners_[:,1])
    range_off = jt.clamp(box_length*1.05, max_v=5.0)
    r_low = polar_corners_[:,1] - range_off
    r_upper = polar_corners_[:,0] + range_off
    
    # In Jittor, boolean indexing works, but assignment needs care. 
    # using jt.where is often safer for conditional updates if direct indexing fails, 
    # but direct indexing is supported in recent versions.
    # Let's try direct indexing first, if issues arise, switch to where.
    
    mask1 = polar_corners_[:, 2] < 0
    mask2 = polar_corners_[:, 3] < 0
    
    # polar_corners_[:, 2][mask1] += 2*np.pi -> This is in-place.
    # Jittor supports in-place via __setitem__ for some cases.
    # If not, use where:
    polar_corners_[:, 2] = jt.where(mask1, polar_corners_[:, 2] + 2*np.pi, polar_corners_[:, 2])
    polar_corners_[:, 3] = jt.where(mask2, polar_corners_[:, 3] + 2*np.pi, polar_corners_[:, 3])

    abs_azi = jt.abs(polar_corners_[:,2] - polar_corners_[:,3])
    azi_mask = abs_azi > (np.pi/2)
    abs_azi = jt.where(azi_mask, 2*np.pi - abs_azi, abs_azi)
    
    box_width =  abs_azi * range_center
    width_off = jt.clamp(box_width*1.05, max_v=5.0)
    azi_off = jt.clamp(width_off / range_center, min_v=0, max_v=0.05)
    
    angle_max = polar_corners_[:, 2] + azi_off
    angle_min = polar_corners_[:, 3] - azi_off
    
    mask3 = angle_max > 2*np.pi
    mask4 = angle_min < 0
    angle_max = jt.where(mask3, angle_max - 2*np.pi, angle_max)
    angle_min = jt.where(mask4, angle_min + 2*np.pi, angle_min)
    
    polar_ranges = polar_pts[:,0].expand(num_box,point_num).transpose(0,1)
    mask0 = polar_pts[:,1] < 0
    polar_pts[:,1] = jt.where(mask0, polar_pts[:,1] + 2*np.pi, polar_pts[:,1])
    
    polar_angles = polar_pts[:,1].expand(num_box,point_num).transpose(0,1)
    
    point_masks_ = (polar_ranges < r_upper) & (polar_ranges > r_low) \
        & (polar_angles > angle_min) & (polar_angles < angle_max)

    return point_masks_


@FUSION_LAYERS.register_module()
class InstanceLevelFusion(nn.Module):
# Proposal-Aware, Point_Gating
    def __init__(self, 
                radii, 
                num_samples, 
                sa_mlps, 
                dilated_group, 
                norm_cfg, 
                sa_cfg, 
                grid_size, 
                grid_fps, 
                code_size,
                num_classes,
                input_channels,
                train_cfg=None,
                test_cfg=None,
                pc_range=None,
                loss_bbox=None,
                loss_cls=None,
                bbox_coder=None):
        super(InstanceLevelFusion, self).__init__()
        self.fp16_enabled = False
        if isinstance(grid_size, (list, tuple)):
            self.grid_size = grid_size[0]
        else:
            self.grid_size = grid_size
        self.grid_fps = grid_fps
        self.code_size = code_size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.pc_range = pc_range
        self.code_weights = [1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 
                            1.0, 1.0, 0.2, 0.2]
        # Jittor doesn't have nn.Parameter exactly like Torch, just use jt.Var or store it.
        # If requires_grad=False, it's just a constant tensor.
        self.code_weights = jt.array(self.code_weights).stop_grad()
        
        self.roi_grid_pool_layer = build_sa_module(
                                    num_point=self.grid_fps,
                                    radii=radii,
                                    sample_nums=num_samples,
                                    mlp_channels=sa_mlps,
                                    dilated_group=dilated_group,
                                    norm_cfg=norm_cfg,
                                    cfg=sa_cfg,
                                    bias=True)
        
        embed_dims = sa_cfg.get('embed_dims', 256) if sa_cfg else 256
        total_sa_channels = sum([mlp[-1] for mlp in sa_mlps])
        
        self.shared_fc_layer = nn.Sequential(
            nn.Conv2d(total_sa_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.reg_layer = nn.Conv1d(512, self.code_size, kernel_size=1, bias=True)
        
        # Training related components skipped or mocked for now if they depend on mmdet
        if train_cfg is not None:
            # self.assigner_refine = build_assigner(train_cfg['assigner'])
            pass

        # sampler_cfg = dict(type='PseudoSampler')
        # self.sampler = build_sampler(sampler_cfg, context=self)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.bbox_coder = build_bbox_coder(bbox_coder)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = True
        
        self.Detr3DCrossAtten = Detr3DCrossAtten(input_dim=total_sa_channels, embed_dims=embed_dims)
        self.channelMixMLPs01 = nn.Sequential(
            nn.Conv1d(6, self.grid_fps//2, kernel_size=1),
            nn.BatchNorm1d(self.grid_fps//2),
            nn.ReLU(), # inplace not needed in jittor
            nn.Conv1d(self.grid_fps//2, self.grid_fps//2, kernel_size=1))
        
        self.linear_p = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=1),
            nn.BatchNorm1d(3),
            nn.ReLU(), 
            nn.Conv1d(3, self.grid_fps//2, kernel_size=1))

        self.channelMixMLPs02 = nn.Sequential(
            nn.Conv1d(self.grid_fps, self.grid_fps//2, kernel_size=1),
            nn.BatchNorm1d(self.grid_fps//2), 
            nn.ReLU(),
            nn.Conv1d(self.grid_fps//2, self.grid_fps//2, kernel_size=1))

        self.channelMixMLPs03 = nn.Conv1d(6, self.grid_fps//2, kernel_size=1)
        # Jittor doesn't have nn.Softmax module, use nn.Softmax function or functional
        # But wait, Jittor HAS nn.Softmax in recent versions? Let's check docs or use functional.
        # Safest is jt.nn.Softmax(dim=-1) if available, or just use ops.
        self.softmax_layer = nn.Softmax(dim=-1)
        
    def execute(self, outs, refine_point, refine_box, img_feats, img_metas):
        # refine_box['boxes'] structure: list of (proposals, head_features)
        # proposals is a Tensor (N, 10) in Torch.
        
        batch_polar_corner = refine_box['polar_corner']
        batch_radar_point = refine_point['original']
        batch_polar_radar_point = refine_point['polar']
        bs = len(batch_polar_radar_point)
        num_box = int(batch_polar_corner.shape[1])
        
        bbox_preds = []
        e_box_indexes = []
        ne_box_indexes = []
        
        for i in range(bs):
            polar_corners = batch_polar_corner[i]
            all_radar_points = batch_radar_point[i]
            polar_radar_points = batch_polar_radar_point[i]
            
            # Assuming these are Jittor Vars now
            head_feature = refine_box['boxes'][i][1].clone() 
            proposals = refine_box['boxes'][i][0]
            if hasattr(proposals, 'tensor'):
                proposals = proposals.tensor
            proposals = proposals.clone()
            
            point_masks = SoftPolarAssociation(polar_radar_points, polar_corners)
            
            # e_box_index: indices where sum of point_masks is 0
            # torch.nonzero().squeeze()
            sum_masks = jt.sum(point_masks, dim=0)
            e_box_index = jt.nonzero(sum_masks == 0)
            if e_box_index.shape[0] > 0:
                e_box_index = e_box_index.squeeze(1) # jt.nonzero returns (N, 1) for 1D
            else:
                e_box_index = jt.array([], dtype='int32') # Empty
                
            ne_box_index = jt.nonzero(sum_masks) # (N, 1)
            # ne_box_index tuple mimic: (ne_box_index_tensor,)
            
            ne_box_num = ne_box_index.shape[0]
            print(f"DEBUG: batch {i}, ne_box_num={ne_box_num}")
            
            if ne_box_num == 0:
                print("DEBUG: ne_box_num is 0, skipping")
                bbox_preds.append([])
                e_box_indexes.append(e_box_index)
                ne_box_indexes.append((ne_box_index,)) # Empty tuple-like
                continue
            elif ne_box_num == 1:
                print("DEBUG: ne_box_num is 1, skipping (logic?)")
                # e_box_index cat ne_box_index[0]
                # In jittor, ne_box_index is (1, 1), we need flatten
                ne_idx_flat = ne_box_index.view(-1)
                e_box_index = jt.concat((e_box_index, ne_idx_flat))
                ne_box_index = (jt.array([], dtype='int32'),)
                bbox_preds.append([])
                e_box_indexes.append(e_box_index)
                ne_box_indexes.append(ne_box_index)
                continue
            else:
                print("DEBUG: ne_box_num > 1, proceeding")
                e_box_indexes.append(e_box_index)
                # Ensure ne_box_index is tuple format expected later? 
                # Torch: ne_box_index is tuple of tensors. Jittor: just tensor usually.
                # Let's keep it as tuple to match Torch logic: ne_box_index[0]
                ne_box_indexes.append((ne_box_index.view(-1),))
            
            # ne_box_index[0] is the indices
            ne_idx = ne_box_index.view(-1)
            ne_proposals = proposals[ne_idx]
            non_empty_pred_box_centers = ne_proposals[:, :3]
            
            ############ Radar Grid Point Generation ############
            print("DEBUG: Generating Radar Grid Points")
            box_2d = ne_proposals[:,:2]
            box_angle = jt.atan2(box_2d[:,1], -box_2d[:,0])
            box_true_vel = ne_proposals[:,-2:]
            box_vel_value = jt.clamp(jt.sqrt((box_true_vel**2).sum(dim=1))/2, min_v=1, max_v=5)
            
            grid_interval = int((self.grid_size-1)/2)
            bin_weight = jt.arange(self.grid_size) - grid_interval
            grid_weight = box_vel_value / grid_interval
            
            roi_box_num = box_2d.shape[0]
            # expand logic: (N, G)
            total_weights_2 = bin_weight.expand(roi_box_num, self.grid_size).transpose(0,1) * grid_weight
            
            vel_x = (jt.cos(box_angle) * total_weights_2).transpose(0,1)
            vel_y = (jt.sin(box_angle) * total_weights_2).transpose(0,1)
            
            # point_masks (N_pts, N_boxes) -> sum axis 0 -> (N_boxes,)
            roi_per_pts_num = sum_masks[ne_idx]
            roi_x_all_pts = point_masks.transpose(0,1)[ne_idx] # (N_boxes, N_pts)
            
            max_pts = int(roi_per_pts_num.max().item())
            print(f"DEBUG: max_pts={max_pts}")
            
            fps_need_box = jt.nonzero(roi_per_pts_num > self.grid_fps//4)
            if fps_need_box.shape[0] > 0: fps_need_box = fps_need_box.squeeze(1)
            fps_num_box = fps_need_box.shape[0]
            
            fps_need_grid_box = jt.nonzero(roi_per_pts_num * self.grid_size > self.grid_fps)
            if fps_need_grid_box.shape[0] > 0: fps_need_grid_box = fps_need_grid_box.squeeze(1)
            fps_num_grid_box = fps_need_grid_box.shape[0]
            print(f"DEBUG: fps_num_box={fps_num_box}, fps_num_grid_box={fps_num_grid_box}")

            radar_points_per_roi = jt.zeros((ne_box_num, self.grid_fps//4, 6))
            grid_points_per_roi = jt.zeros((ne_box_num, self.grid_fps, 3))
            
            if fps_num_box > 0 :
                batch_cur_pts = jt.zeros((fps_num_box, max_pts, 6))
            if fps_num_grid_box > 0 :
                batch_cur_grid_pts = jt.zeros((fps_num_grid_box, max_pts*self.grid_size, 6))
            
            for box_idx in range(ne_box_num):
                # roi_x_all_pts[box_idx] is boolean mask
                pts_index = jt.nonzero(roi_x_all_pts[box_idx]).view(-1)
                
                # Check if box_idx in fps_need_box
                # In Jittor, simple 'in' might not work on Vars. Use mask check.
                is_in_fps_box = (fps_need_box == box_idx).any()
                
                if not is_in_fps_box:
                    # Random sampling
                    needed = self.grid_fps//4 - pts_index.shape[0]
                    if needed > 0:
                        random_idx = jt.randint(0, pts_index.shape[0], (needed,))
                        random_idx = jt.concat((jt.arange(pts_index.shape[0]), random_idx))
                    else:
                        random_idx = jt.arange(pts_index.shape[0])[:self.grid_fps//4] # Should not happen based on logic but safety
                        
                    cur_roi_point_xyz = all_radar_points[pts_index[random_idx]]
                    radar_points_per_roi[box_idx] = cur_roi_point_xyz
                else:
                    new_idx = jt.nonzero(fps_need_box == box_idx).item()
                    needed = max_pts - pts_index.shape[0]
                    if needed > 0:
                        random_idx = jt.randint(0, pts_index.shape[0], (needed,))
                        random_idx = jt.concat((jt.arange(pts_index.shape[0]), random_idx))
                    else:
                        random_idx = jt.arange(max_pts) # Should match
                        
                    cur_roi_point_xyz = all_radar_points[pts_index[random_idx]]
                    batch_cur_pts[new_idx] = cur_roi_point_xyz
                
                # Grid points generation
                # grid_pts: (N_pts * grid_size, 6)
                grid_pts = all_radar_points[pts_index].unsqueeze(0).expand(self.grid_size, pts_index.shape[0], 6).reshape(-1, 6).clone()
                
                # Add velocity offsets
                # vel_x[box_idx]: (grid_size,)
                vx = vel_x[box_idx].unsqueeze(0).unsqueeze(2).expand(pts_index.shape[0], self.grid_size, 1).reshape(-1, 1)
                vy = vel_y[box_idx].unsqueeze(0).unsqueeze(2).expand(pts_index.shape[0], self.grid_size, 1).reshape(-1, 1)
                
                # Add offsets to x and y
                # grid_pts[:, 0] += vx[:, 0]
                # grid_pts[:, 1] += vy[:, 0]
                grid_pts[:, 0] = grid_pts[:, 0] + vx[:, 0]
                grid_pts[:, 1] = grid_pts[:, 1] + vy[:, 0]
                
                # Update velocity
                # grid_pts[:, 3] = vx[:, 0]
                # grid_pts[:, 4] = vy[:, 0]
                grid_pts[:, 3] = vx[:, 0]
                grid_pts[:, 4] = vy[:, 0]
                
                is_in_fps_grid_box = (fps_need_grid_box == box_idx).any()
                
                if not is_in_fps_grid_box:
                    needed = self.grid_fps - grid_pts.shape[0]
                    if needed > 0:
                        random_idx = jt.randint(0, grid_pts.shape[0], (needed,))
                        random_idx = jt.concat((jt.arange(grid_pts.shape[0]), random_idx))
                    else:
                        random_idx = jt.arange(grid_pts.shape[0])[:self.grid_fps]
                    
                    cur_roi_grid_point_xyz = grid_pts[random_idx, :3]
                    grid_points_per_roi[box_idx] = cur_roi_grid_point_xyz
                else:
                    new_idx = jt.nonzero(fps_need_grid_box == box_idx).item()
                    needed = max_pts*self.grid_size - grid_pts.shape[0]
                    if needed > 0:
                        random_idx = jt.randint(0, grid_pts.shape[0], (needed,))
                        random_idx = jt.concat((jt.arange(grid_pts.shape[0]), random_idx))
                    else:
                        random_idx = jt.arange(max_pts*self.grid_size)
                        
                    cur_roi_grid_point_xyz = grid_pts[random_idx, :3]
                    # This logic seems complicated for Jittor memory
                    # batch_cur_grid_pts[new_idx] = cur_roi_grid_point_xyz # Wait, size mismatch?
                    # batch_cur_grid_pts is (fps_num_grid_box, max_pts*self.grid_size, 6)
                    # grid_pts is (..., 6)
                    # But we only assign xyz to grid_points_per_roi?
                    # Original code logic:
                    # grid_points_per_roi[box_idx] = grid_pts[random_idx, :3] # Only xyz
                    
                    # For batch_cur_grid_pts, it stores all 6 dims?
                    batch_cur_grid_pts[new_idx] = grid_pts[random_idx]

            print(f"DEBUG: Loop finished for batch {i}")

            if fps_num_box > 0 :
                # furthest_point_sample returns indices
                fps_pts_idx = furthest_point_sample(batch_cur_pts[..., :3], self.grid_fps//4)
                # Jittor indexing: batch_cur_pts[batch_indices, fps_pts_idx]
                # Need to manually handle batch indexing if not supported directly
                for fps_box in range(fps_num_box):
                    idx = fps_pts_idx[fps_box] # (K,)
                    fps_cur_pts = batch_cur_pts[fps_box][idx]
                    radar_points_per_roi[fps_need_box[fps_box]] = fps_cur_pts
            
            if fps_num_grid_box > 0 :
                fps_grid_idx = furthest_point_sample(batch_cur_grid_pts[..., :3], self.grid_fps)
                for fps_grid_box in range(fps_num_grid_box):
                    idx = fps_grid_idx[fps_grid_box]
                    fps_cur_grid_pts = batch_cur_grid_pts[fps_grid_box][idx][..., :3]
                    grid_points_per_roi[fps_need_grid_box[fps_grid_box]] = fps_cur_grid_pts
            
            ############ proposal-aware radar attention ############
            # relative_pos: (N_box, N_pts, 3)
            # non_empty_pred_box_centers: (N_box, 3) -> expand
            # radar_points_per_roi: (N_box, N_pts, 6)
            
            relative_pos = jt.abs(non_empty_pred_box_centers.unsqueeze(1).expand(ne_box_num, self.grid_fps//4, 3) - radar_points_per_roi[..., :3])
            
            # channelMixMLPs01 expects (B, C, N)
            energy = self.channelMixMLPs01(radar_points_per_roi.permute(0,2,1))
            p_embed = self.linear_p(relative_pos.permute(0,2,1))
            
            energy = jt.concat([energy, p_embed], dim=1)
            energy = self.channelMixMLPs02(energy)
            w = self.softmax_layer(energy)
            
            x_v = self.channelMixMLPs03(radar_points_per_roi.permute(0,2,1))
            radar_attn_feat = ((x_v + p_embed) * w).permute(0,2,1)
            
            ############ Radar Grid Point Pooling ############
            # grid_points_per_roi_: (N_box * N_grid, 3)
            grid_points_per_roi_ = grid_points_per_roi.view(-1, 3)
            
            # roi_grid_pool_layer (SA module)
            # points_xyz: (1, N_box*N_pts, 3)
            # features: (1, C, N_box*N_pts)
            # new_xyz: (1, N_box*N_grid, 3)
            
            # Fix for channel mismatch: 
            # radar_attn_feat has 512 channels, but SA module expects 32.
            # We treat this as 16 feature vectors per point.
            # So we must repeat points_xyz 16 times to match.
            
            C_feat = radar_attn_feat.shape[2]
            C_req = 32
            factor = C_feat // C_req
            
            # Prepare features: [ne_box_num, 256, 512] -> [ne_box_num, 256, 16, 32] -> flattened
            features_in = radar_attn_feat.view(ne_box_num, -1, factor, 32).view(-1, 32).unsqueeze(0).permute(0,2,1)
            
            # Prepare points: [ne_box_num, 256, 3] -> [ne_box_num, 256, 16, 3] -> flattened
            points_xyz_in = radar_points_per_roi[..., :3].unsqueeze(2).repeat(1, 1, factor, 1).view(-1, 3).unsqueeze(0)
            
            print(f"DEBUG: radar_attn_feat shape: {radar_attn_feat.shape}")
            print(f"DEBUG: features_in shape: {features_in.shape}")
            print(f"DEBUG: points_xyz_in shape: {points_xyz_in.shape}")
            
            pooled_point, pooled_feature = self.roi_grid_pool_layer(
                points_xyz = points_xyz_in,
                features = features_in,
                new_xyz = grid_points_per_roi_.unsqueeze(0)
            )
            
            ### Detr3D style image feature pooling ###
            # img_feats is list of feature maps
            value=[]
            for mlvl_feat_num in range(len(img_feats)):
                value.append(img_feats[mlvl_feat_num][i].unsqueeze(0))
            
            query = pooled_feature.permute(2, 0, 1) # (N, B, C)
            key = None
            reference_points = pooled_point.clone()
            
            # Detr3DCrossAtten Jittor version
            sampled_feat = self.Detr3DCrossAtten(query, key, value, reference_points=reference_points, img_metas=img_metas[i])
            
            # sampled_feat: (N_query, B, C) -> permute -> (B, C, N_query)
            # N_query = ne_box_num * self.grid_fps
            
            sampled_feats = sampled_feat.permute(1, 2, 0).view(1, -1, ne_box_num, self.grid_fps)
            # (1, C, ne_box_num, grid_fps)
            
            # shared_fc_layer expects (B, C, H, W) ? 
            # Torch: shared_fc_layer(sampled_feats).max(-1)[0]
            # sampled_feats shape is (1, C, ne_box_num, grid_fps) -> treated as (B, C, H, W) where H=ne_box_num, W=grid_fps
            
            shared_features = self.shared_fc_layer(sampled_feats)
            shared_features = jt.max(shared_features, dim=-1) # (1, 256, ne_box_num)
            
            # head_feature_ = head_feature[ne_idx].unsqueeze(0).transpose(1, 2) # (1, C, ne_box_num)
            
            hf_subset = head_feature[ne_idx]
            print(f"DEBUG: head_feature shape: {head_feature.shape}")
            print(f"DEBUG: ne_idx shape: {ne_idx.shape}")
            print(f"DEBUG: hf_subset shape: {hf_subset.shape}")
            
            head_feature_ = hf_subset.unsqueeze(0).permute(0, 2, 1)
            print(f"DEBUG: head_feature_ shape: {head_feature_.shape}")
            
            refine_feature = jt.concat((head_feature_, shared_features), dim=1)
            
            bbox_pred = self.reg_layer(refine_feature).transpose(1, 2).squeeze(0) # (ne_box_num, 10)
            bbox_preds.append(bbox_pred)
            
        # Post-processing bbox_preds to outs['all_bbox_preds_refine']
        # Fix: Determine box dimension from input proposals and handle batch loop correctly
        
        # Get dimension from first available proposal tensor
        sample_proposals = refine_box['boxes'][0][0]
        if hasattr(sample_proposals, 'tensor'):
            sample_proposals = sample_proposals.tensor
        input_dim = sample_proposals.shape[-1]
        
        # Ensure we have at least 10 dims for the update logic if required, 
        # or adapt update logic. The code below uses index 8:10 which implies 10 dims.
        # If input is 9, we might need to pad or adjust indices.
        # For safety, let's use max(input_dim, 10) for box_reg, but we need to match input.
        
        # Let's trust input_dim for box_reg, but handle update carefully.
        box_reg = jt.zeros((bs, num_box, input_dim))
        
        for i in range(bs):
            # Fix: Fetch proposals for the CURRENT batch index i
            proposals = refine_box['boxes'][i][0]
            if hasattr(proposals, 'tensor'):
                proposals = proposals.tensor
            
            # Ensure proposals is Jittor var
            if not isinstance(proposals, jt.Var):
                proposals = jt.array(proposals)
                
            if len(e_box_indexes[i]) > 0:
                e_boxes_ori = proposals[e_box_indexes[i]]
                box_reg[i, e_box_indexes[i]] = e_boxes_ori
                
            if len(ne_box_indexes[i][0]) > 0:
                ne_idx = ne_box_indexes[i][0]
                ne_boxes_ori = proposals[ne_idx]
                
                # Apply offsets
                preds = bbox_preds[i]
                # preds shape: (ne_box_num, 10)
                
                # Create refined boxes
                refined_boxes = ne_boxes_ori.clone()
                
                # Update x, y (0:2)
                refined_boxes[..., 0:2] = ne_boxes_ori[..., 0:2] + preds[..., 0:2]
                
                # Update w, l, h (3:6) -> assuming indices 3,4,5
                # preds indices 2:5
                refined_boxes[..., 3:6] = ne_boxes_ori[..., 3:6] + preds[..., 2:5]
                
                # Update velocity? 
                # Original code: ne_boxes_ori[..., 8:10] += preds[..., 5:7]
                # If input_dim is 9, indices are 0..8. 8:10 is slice [8].
                # If input_dim is 9, velocity is usually 7:9.
                
                if input_dim >= 10:
                    refined_boxes[..., 8:10] = ne_boxes_ori[..., 8:10] + preds[..., 5:7]
                elif input_dim == 9:
                    # Assume velocity is 7:9
                    refined_boxes[..., 7:9] = ne_boxes_ori[..., 7:9] + preds[..., 5:7]
                
                box_reg[i, ne_idx] = refined_boxes
                
        box_reg = box_reg.view(1, bs, num_box, input_dim)
        outs['all_bbox_preds_refine'] = box_reg
        return outs
