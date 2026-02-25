
import jittor as jt
from jittor import nn
import numpy as np
import copy
import warnings
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from projects.mmdet3d_plugin.rcm_fusion.modules.radar_camera_gating import RadarCameraGating
from projects.mmdet3d_plugin.jittor_adapter import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER, build_from_cfg
from jittor.nn import Module as BaseModule

def build_transformer_layer(cfg):
    return build_from_cfg(cfg, TRANSFORMER_LAYER)

def force_fp32(apply_to=None):
    def decorator(func):
        return func
    return decorator

def auto_fp16(apply_to=None):
    def decorator(func):
        return func
    return decorator

@TRANSFORMER_LAYER.register_module()
class RadarGuidedBEVEncoderLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=None,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 ffn_cfgs=None,
                 **kwargs):
        
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=feedforward_channels if feedforward_channels is not None else 1024,
                num_fcs=ffn_num_fcs,
                ffn_drop=ffn_dropout,
                act_cfg=act_cfg,
            )
            
        super(RadarGuidedBEVEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        self.fp16_enabled = False
        self.radar_camera_gating = RadarCameraGating()
        self.pts_adaptive_layer = nn.Sequential(
            nn.Conv1d(256,256,kernel_size=1,padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # assert len(operation_order) == 7
        # Force recompile
        pass
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def execute(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                pts_bev=None,
                **kwargs):
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, jt.Var):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        
        bs = query.shape[0]
        # Fix: Ensure bev_h and bev_w are treated as correct values when creating tensor
        # jt.array([[bev_h, bev_w]]) resulting in 0s suggests type inference issue
        # Use numpy as intermediate to ensure correctness
        bev_spatial_shapes_np = np.array([[bev_h, bev_w]], dtype=np.int32)
        bev_spatial_shapes = jt.array(bev_spatial_shapes_np)
        print(f"DEBUG: RadarGuidedBEVEncoderLayer spatial_shapes (input): {spatial_shapes.shape}, data: {spatial_shapes.numpy()}")
        
        # level_start_index = jt.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        bev_level_start_index = jt.zeros((1,), dtype=jt.int32)

        for layer in self.operation_order:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    pts_bev,
                    pts_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=bev_spatial_shapes,
                    level_start_index=bev_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                print(f"DEBUG: Calling cross_attn. spatial_shapes: {spatial_shapes.shape}, data: {spatial_shapes.numpy()}")
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query
            
            elif layer == 'ffn':
                radar_bev = self.pts_adaptive_layer(pts_bev[0:bs].permute(0,2,1)).permute(0,2,1)
                query = self.radar_camera_gating(query, radar_bev)
                identity = query
                query = self.norms[-1](query)

                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        return query

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RadarGuidedBEVEncoder(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=None, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):
        super(RadarGuidedBEVEncoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        
        if isinstance(transformerlayers, dict):
            layer_cfgs = [transformerlayers for _ in range(num_layers)]
        elif isinstance(transformerlayers, list):
            layer_cfgs = transformerlayers
        else:
            layer_cfgs = []
            
        self.layers = nn.ModuleList()
        for cfg in layer_cfgs:
            self.layers.append(build_transformer_layer(cfg))

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype='float32'):
        if dim == '3d':
            zs = jt.linspace(0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = jt.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = jt.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = jt.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            y_lin = jt.linspace(0.5, H - 0.5, H)
            x_lin = jt.linspace(0.5, W - 0.5, W)
            ref_y, ref_x = jt.meshgrid(y_lin, x_lin)
            ref_y = ref_y.reshape(-1).unsqueeze(0) / H
            ref_x = ref_x.reshape(-1).unsqueeze(0) / W
            ref_2d = jt.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        print(f"DEBUG: point_sampling start")
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = jt.array(lidar2img)
        print(f"DEBUG: lidar2img initial shape: {lidar2img.shape}, numel: {lidar2img.numel()}")

        reference_points = reference_points.clone()
        print(f"DEBUG: reference_points initial shape: {reference_points.shape}")

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = jt.concat(
            (reference_points, jt.ones_like(reference_points[..., :1])), -1)
        print(f"DEBUG: reference_points after concat shape: {reference_points.shape}")
        
        reference_points = reference_points.permute(1, 0, 2, 3)
        print(f"DEBUG: reference_points after permute shape: {reference_points.shape}")
        
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]
        
        print(f"DEBUG: D={D}, B={B}, num_query={num_query}, num_cam={num_cam}")
        
        print(f"DEBUG: Reshaping reference_points. Target view: ({D}, {B}, 1, {num_query}, 4)")
        rp_view = reference_points.view(D, B, 1, num_query, 4)
        print(f"DEBUG: rp_view shape: {rp_view.shape}")
        reference_points = rp_view.repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        print(f"DEBUG: reference_points final shape: {reference_points.shape}")

        print(f"DEBUG: Reshaping lidar2img. Target view: (1, {B}, {num_cam}, 1, 4, 4)")
        expected_lidar_numel = 1 * B * num_cam * 1 * 4 * 4
        if lidar2img.numel() != expected_lidar_numel:
            print(f"ERROR: lidar2img numel {lidar2img.numel()} != expected {expected_lidar_numel}")
            
        l2i_view = lidar2img.view(
            1, B, num_cam, 1, 4, 4)
        print(f"DEBUG: l2i_view shape: {l2i_view.shape}")
        lidar2img = l2i_view.repeat(D, 1, 1, num_query, 1, 1)
        print(f"DEBUG: lidar2img final shape: {lidar2img.shape}")

        reference_points_cam = jt.matmul(lidar2img, reference_points).squeeze(-1)
        
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / jt.maximum(
            reference_points_cam[..., 2:3], jt.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def execute(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                pts_bev=None,
                shift=0.,
                **kwargs):
        print(f"DEBUG: RadarGuidedBEVEncoder execute. bev_query shape: {bev_query.shape}")
        output = bev_query
        intermediate = []
        
        # Jittor Fix: bs should be shape[0] because input is batch_first (BS, Len, Dim)
        bs = bev_query.shape[0]

        # Jittor Fix: Ensure pts_bev is batch_first [BS, Len, Dim] if it is [Len, BS, Dim]
        if pts_bev is not None and pts_bev.ndim == 3 and pts_bev.shape[1] == bs and pts_bev.shape[0] != bs:
             pts_bev = pts_bev.permute(1, 0, 2)
             print(f"DEBUG: pts_bev permuted to {pts_bev.shape}")
        
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bs)
        
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # Removed permutations to maintain batch_first
        # bev_query = bev_query.permute(1, 0, 2)
        # bev_pos = bev_pos.permute(1, 0, 2)
        
        # Simplified ref_2d and prev_bev handling (removed doubling of BS)
        # bs, len_bev, num_bev_level, _ = ref_2d.shape
        
        # if prev_bev is not None:
        #     prev_bev = prev_bev.permute(1, 0, 2)
        #     prev_bev = jt.stack(
        #         [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
        #     hybird_ref_2d = jt.stack([shift_ref_2d, ref_2d], 1).reshape(
        #         bs*2, len_bev, num_bev_level, 2)
        # else:
        #     hybird_ref_2d = jt.stack([ref_2d, ref_2d], 1).reshape(
        #         bs*2, len_bev, num_bev_level, 2)
        
        # if pts_bev is not None:
        #     pts_bev = pts_bev.permute(1, 0, 2)
        #     pts_bev = jt.stack(
        #         [pts_bev, bev_query], 1).reshape(bs*2, len_bev, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                output, # Use output as query for next layer
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                pts_bev=pts_bev,
                **kwargs)

            # bev_query = output # Update bev_query? No, output is enough.
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return jt.stack(intermediate)

        return output
