
import copy
import warnings
import jittor as jt
import jittor.nn as nn
import math
import numpy as np

from projects.mmdet3d_plugin.jittor_adapter import (
    ATTENTION, TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER,
    build_transformer_layer,
    build_activation_layer,
    build_norm_layer
)
from projects.mmdet3d_plugin.rcm_fusion.modules.custom_base_transformer_layer import BaseModule, MyCustomBaseTransformerLayer
from projects.mmdet3d_plugin.rcm_fusion.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

print(f"Loading decoder.py, TRANSFORMER_LAYER id: {id(TRANSFORMER_LAYER)}")

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Var): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Var: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(0, 1)
    x1 = x.clamp(eps, 1.0)
    x2 = (1 - x).clamp(eps, 1.0)
    return jt.log(x1 / x2)

class TransformerLayerSequence(BaseModule):
    """Base class for TransformerLayerSequence."""
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequence, self).__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list)
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))

    def execute(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, all the attention modules in `operation_order` will
            be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for FFN, the order should be consistent with it in
            `operation_order`. If it is a dict, all the FFN modules in
            `operation_order` will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs=None,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='ReLU', inplace=True),
            )
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            batch_first=batch_first,
            **kwargs)

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def execute(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Var): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Var): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Var: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            # output = output.permute(1, 0, 2)

            print(f"DEBUG: Decoder loop {lid}, output.shape={output.shape}")

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                print(f"DEBUG: Decoder loop {lid}, tmp.shape={tmp.shape}, reference_points.shape={reference_points.shape}")
                
                print("DEBUG: Checking reference_points values...")
                if jt.any(jt.isnan(reference_points)):
                     print("DEBUG: reference_points has NaNs!")
                if jt.any(jt.isinf(reference_points)):
                     print("DEBUG: reference_points has Infs!")

                if reference_points.shape[-1] == 4:
                    new_reference_points = reference_points + tmp
                elif reference_points.shape[-1] == 2:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = jt.zeros_like(reference_points)
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points[..., :2])
                    # No z-coordinate update for 2D reference points
                elif reference_points.shape[-1] == 3:
                    assert reference_points.shape[-1] == 3
                    
                    print("DEBUG: Processing reference_points 3D. Before inverse_sigmoid.")
                    xy_inv = inverse_sigmoid(reference_points[..., :2])
                    print("DEBUG: inverse_sigmoid(xy) done.")
                    z_inv = inverse_sigmoid(reference_points[..., 2:3])
                    print("DEBUG: inverse_sigmoid(z) done.")

                    new_reference_points = jt.zeros_like(reference_points)
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + xy_inv
                    new_reference_points[..., 2:3] = tmp[
                        ..., 4:5] + z_inv

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return jt.stack(intermediate), jt.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.sampling_offsets, 0.)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)

        thetas = jt.arange(
            self.num_heads,
            dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdims=True)).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias = grid_init.view(-1)
        
        # constant_init(self.attention_weights, val=0., bias=0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

        # xavier_init(self.value_proj, distribution='uniform', bias=0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)

        # xavier_init(self.output_proj, distribution='uniform', bias=0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
        
        self._is_init = True

    def execute(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Var): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Var): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Var): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Var): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Var): The positional encoding for `query`.
                Default: None.
            key_pos (Var): The positional encoding for `key`. Default
                None.
            reference_points (Var):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Var): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Var): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Var): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Var: forwarded results with shape [num_query, bs, embed_dims].
        """
        if value is None:
            value = query
        
        print(f"DEBUG: Decoder.execute start. self.batch_first={self.batch_first}")
        print(f"DEBUG: Input query.shape={query.shape}")
        if value is not None:
             print(f"DEBUG: Input value.shape={value.shape}")

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        if value.ndim == 4:
            value = value.squeeze(2)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = jt.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        # Jittor doesn't have is_cuda check in same way, and we use one implementation for now
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
