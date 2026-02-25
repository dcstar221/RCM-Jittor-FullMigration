# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jisong Kim and Minjae Seong
# ---------------------------------------------

import numpy as np
import jittor as jt
from jittor import nn
from jittor.init import xavier_uniform_, gauss_ as normal_
# from mmcv.cnn import xavier_init
# from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from jittor.nn import Module as BaseModule

from projects.mmdet3d_plugin.jittor_adapter import TRANSFORMER, build_transformer_layer_sequence
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
# from torchvision.transforms.functional import rotate
from .radar_guided_bev_attention import RadarGuidedBEVAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
# from mmcv.runner import force_fp32, auto_fp16

def auto_fp16(apply_to=None, out_fp32=False):
    def decorator(func):
        return func
    return decorator

def rotate(img, angle, center=None):
    # img: (C, H, W)
    # angle: degrees
    # center: (x, y) - currently assumed to be image center if not handled explicitly
    
    if isinstance(angle, (int, float)) and angle == 0:
        return img

    # Jittor implementation using grid_sample
    C, H, W = img.shape
    
    # angle is in degrees, convert to radians
    theta = -angle * np.pi / 180.0 # Inverse rotation for grid sampling
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Affine matrix [2, 3]
    # x_src = x_dst * cos - y_dst * sin
    # y_src = x_dst * sin + y_dst * cos
    # We need to construct the grid
    
    # Create grid of coordinates [-1, 1]
    # shape: (H, W, 2)
    
    # It's faster to do this:
    # theta_mat = [[cos, -sin, 0], [sin, cos, 0]]
    theta_mat = jt.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0]], dtype=img.dtype)
    theta_mat = theta_mat.unsqueeze(0) # (1, 2, 3)
    
    # Use affine_grid if available or construct manually
    # Jittor doesn't strictly have affine_grid in public API sometimes, let's use manual grid if simple
    # But usually jt.nn.affine_grid should work if similar to torch
    # Let's try to construct grid manually to be safe and clear
    
    xx = jt.linspace(-1, 1, W)
    yy = jt.linspace(-1, 1, H)
    grid_y, grid_x = jt.meshgrid(yy, xx) # (H, W)
    grid = jt.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
    grid = grid.unsqueeze(0) # (1, H, W, 2)
    
    # Apply rotation
    # grid is (N, H, W, 2)
    # theta is (N, 2, 3)
    # We want (N, H, W, 2)
    
    # x_new = x * cos + y * (-sin) + 0
    # y_new = x * sin + y * cos + 0
    
    # Reshape grid to (1, H*W, 2)
    # Multiply with theta (2, 2 part)
    
    # Simplified:
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    
    x = grid[..., 0]
    y = grid[..., 1]
    
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    
    grid_new = jt.stack([x_new, y_new], dim=-1)
    
    # Sample
    # img is (C, H, W) -> need (1, C, H, W)
    img_batched = img.unsqueeze(0)
    
    # grid_sample expects (N, H, W, 2)
    # align_corners=False is usually default in some libs, but let's stick to Jittor default
    # padding_mode='zeros'
    
    out = nn.grid_sample(img_batched, grid_new, align_corners=False, padding_mode='zeros')
    
    return out.squeeze(0)

@TRANSFORMER.register_module()
class PerceptionTransformerRadar(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformerRadar, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = jt.zeros((self.num_feature_levels, self.embed_dims))
        self.cams_embeds = jt.zeros((self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        # Jittor initializes vars automatically, but we can re-init
        for p in self.parameters():
            if p.ndim > 1:
                xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MSDeformableAttention3D, RadarGuidedBEVAttention, CustomMSDeformableAttention)):
                try:
                    m.init_weight()
                except AttributeError:
                    if hasattr(m, 'init_weights'):
                        m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_uniform_(self.reference_points.weight)
        if self.reference_points.bias is not None:
            jt.init.constant_(self.reference_points.bias, 0.)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.) 
        # Manual init for Sequential is tricky in Jittor loop, but default is usually fine or we can iterate
        for m in self.can_bus_mlp.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos', 'pts_bev'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            pts_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        
        bs = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = jt.array(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                rotated_prev_bevs = []
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, -1)
                    rotated_prev_bevs.append(tmp_prev_bev)
                prev_bev = jt.stack(rotated_prev_bevs, dim=1)
        
        if pts_bev is not None:
            # check the shape of pts_bev at SECONDFPN_v2
            # assert pts_bev.shape[1] == bs
            print(f"DEBUG: pts_bev check passed. pts_bev.shape={pts_bev.shape}")
            if len(pts_bev.shape) == 4:
                pts_bev = pts_bev.flatten(2).permute(2, 0, 1)

        # add can bus signals
        can_bus = jt.array(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                # feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.cams_embeds[:, None, None, :]
            # feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = jt.concat(feat_flatten, 2)
        print(f"DEBUG: transformer_radar feat_flatten shape: {feat_flatten.shape}")
        # spatial_shapes = jt.array(spatial_shapes) # , dtype=torch.long, device=bev_pos.device)
        
        # Use numpy for safe creation
        spatial_shapes_np = np.array(spatial_shapes, dtype=np.int32)
        spatial_shapes = jt.array(spatial_shapes_np)
        print(f"DEBUG: transformer_radar spatial_shapes: {spatial_shapes.shape}, data: {spatial_shapes.numpy()}")
        
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        level_start_index = jt.concat((jt.zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # feat_flatten = feat_flatten.permute(
        #    0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        
        # Change to (bs, num_cam, H*W, embed_dims) for compatibility with RadarGuidedBEVAttention
        feat_flatten = feat_flatten.permute(1, 0, 2, 3)
        
        print("DEBUG: Before calling encoder in get_bev_features")
        print(f"DEBUG: bev_queries shape: {bev_queries.shape}")
        
        # Expand bev_queries to (Len, BS, Dim)
        # bev_queries is (Len, Dim)
        if bev_queries.ndim == 2:
             bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        
        # Ensure batch_first=True for encoder
        bev_queries = bev_queries.permute(1, 0, 2)
        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)
        
        print(f"DEBUG: bev_queries expanded shape: {bev_queries.shape}")

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            pts_bev=pts_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'pts_bev', 'bev_pos'))
    def execute(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                pts_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`."""

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            pts_bev=pts_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        # query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos, query = jt.split(object_query_embed, self.embed_dims, dim=1)
        
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # [num_query, bs, dim] -> [bs, num_query, dim]
        # query = query.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        # bev_embed = bev_embed.permute(1, 0, 2)
        # breakpoint()
        # Fix: Ensure bev_h and bev_w are treated as correct values when creating tensor
        # Use numpy as intermediate to ensure correctness
        spatial_shapes_np = np.array([[bev_h, bev_w]], dtype=np.int32)
        spatial_shapes = jt.array(spatial_shapes_np)

        # Ensure batch_first=True for decoder if needed
        # CustomMSDeformableAttention in decoder might expect batch_first=True
        
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=jt.array([0]),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
