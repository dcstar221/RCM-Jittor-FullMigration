
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from projects.mmdet3d_plugin.jittor_adapter import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
    build_transformer_layer_sequence
)
# from mmcv.cnn.bricks.transformer import TransformerLayerSequence # Removed
import jittor as jt
from jittor import nn
import numpy as np
import cv2 as cv

# Jittor doesn't have a direct equivalent to TransformerLayerSequence in public API often, 
# but we can define a simple one or use the one from adapter if available.
# Checking adapter, it registers module but might not provide base class.
# We will define a BaseTransformerLayerSequence here or assume MyCustomBaseTransformerLayer is enough?
# No, Sequence is different.
# Let's define a minimal TransformerLayerSequence compatible with Jittor.

class TransformerLayerSequence(nn.Module):
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformerlayers, dict):
            transformerlayers = [copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
        
        from projects.mmdet3d_plugin.jittor_adapter import build_transformer_layer
        for cfg in transformerlayers:
            self.layers.append(build_transformer_layer(cfg))

    def init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p)

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device=None, dtype=None):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            # zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
            #                     device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            zs = jt.linspace(0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            
            # xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
            #                     device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            xs = jt.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            
            # ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
            #                     device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ys = jt.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            
            ref_3d = jt.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            ref_3d = ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            # ref_y, ref_x = torch.meshgrid(
            #     torch.linspace(
            #         0.5, H - 0.5, H, dtype=dtype, device=device),
            #     torch.linspace(
            #         0.5, W - 0.5, W, dtype=dtype, device=device)
            # )
            # Jittor meshgrid order might be different. 
            # numpy meshgrid(x, y, indexing='xy') -> (X, Y) where X varies along columns, Y along rows.
            # torch meshgrid(x, y, indexing='ij') -> (X, Y) where X varies along rows, Y along columns.
            
            # We want ref_y (H, W) where values vary along H (rows).
            # We want ref_x (H, W) where values vary along W (cols).
            
            yy = jt.linspace(0.5, H - 0.5, H)
            xx = jt.linspace(0.5, W - 0.5, W)
            
            # jt.meshgrid(yy, xx) -> (Y, X)
            # Y[i, j] = yy[i] -> varies along rows (H)
            # X[i, j] = xx[j] -> varies along cols (W)
            # This matches torch 'ij' indexing which is what we want for ref_y, ref_x
            
            ref_y, ref_x = jt.meshgrid(yy, xx)
            
            ref_y = ref_y.reshape(-1).unsqueeze(0) / H
            ref_x = ref_x.reshape(-1).unsqueeze(0) / W
            ref_2d = jt.stack((ref_x, ref_y), -1)
            # ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    # @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        lidar2img = jt.array(lidar2img)
        # reference_points = reference_points.clone() # Jittor tensors are not mutable in place usually

        # reference_points[..., 0:1] = reference_points[..., 0:1] * \
        #     (pc_range[3] - pc_range[0]) + pc_range[0]
        # reference_points[..., 1:2] = reference_points[..., 1:2] * \
        #     (pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_points[..., 2:3] = reference_points[..., 2:3] * \
        #     (pc_range[5] - pc_range[2]) + pc_range[2]
        
        # Jittor doesn't support inplace slice update easily like x[..., 0:1] = ...
        # We need to reconstruct or use specific ops.
        
        x = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        y = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        z = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        
        # reference_points = torch.cat(
        #     (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        
        reference_points = jt.concat([x, y, z, jt.ones_like(x)], dim=-1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
        #                                     reference_points.to(torch.float32)).squeeze(-1)
        reference_points_cam = jt.matmul(lidar2img, reference_points).squeeze(-1)
        
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        # reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        #     reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        denom = jt.maximum(reference_points_cam[..., 2:3], jt.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        
        # Note: Inplace update again.
        
        # Calculate normalized coords
        res_x = (reference_points_cam[..., 0:1] / denom) / img_metas[0]['img_shape'][0][1]
        res_y = (reference_points_cam[..., 1:2] / denom) / img_metas[0]['img_shape'][0][0]
        
        # Reconstruct reference_points_cam with updated x, y and original z (or z is not used anymore?)
        # Original code kept z for mask check but here we already computed mask.
        # But we need z?
        # Original code: reference_points_cam[..., 0:2] / ...
        # It kept z at index 2.
        
        res_z = reference_points_cam[..., 2:3] # Keep original Z or divided Z?
        # Original: reference_points_cam = reference_points_cam[..., 0:2] / max(...)
        # This replaces the first 2 channels. It returns a tensor of shape (..., 2)? 
        # No, "reference_points_cam[..., 0:2] / ..." calculates a (..., 2) tensor.
        # But "reference_points_cam = ..." assigns it to variable.
        # Wait, if it assigns to variable, the variable now has 2 channels.
        # BUT later: "bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0) ..."
        # So reference_points_cam must have at least 2 channels.
        
        reference_points_cam = jt.concat([res_x, res_y], dim=-1)

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        #     bev_mask = torch.nan_to_num(bev_mask)
        # else:
        #     bev_mask = bev_mask.new_tensor(
        #         np.nan_to_num(bev_mask.cpu().numpy()))
        
        # Jittor nan_to_num? 
        # jt.where(jt.isnan(bev_mask), 0, bev_mask)
        # But bev_mask is boolean (0 or 1). It shouldn't have NaNs unless inputs had NaNs.
        # If inputs had NaNs, comparison results might be 0.
        # But let's be safe.
        # bev_mask is float/int in Jittor if used in arithmetic.
        
        # Ensure float for nan check if needed, but for bool mask?
        # Let's assume it's fine or convert to float.
        bev_mask = bev_mask.float()
        bev_mask = jt.where(jt.isnan(bev_mask), jt.zeros_like(bev_mask), bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    # @auto_fp16()
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
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.shape[1],  device=None, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.shape[1], device=None, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        
        # shift += shift[:, None, None, :] 
        # shift is tensor?
        # shift += ... might be inplace.
        # shift_ref_2d += shift[:, None, None, :]
        
        # Assuming shift is compatible shape
        if isinstance(shift, (int, float)):
             pass
        else:
             shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = jt.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = jt.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = jt.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return jt.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
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
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

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
        
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=jt.array(
                        [[bev_h, bev_w]]),
                    level_start_index=jt.array([0]),
                    **kwargs)
                
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
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
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
