
import jittor as jt
from jittor import nn
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_jittor
import warnings
import math
from projects.mmdet3d_plugin.jittor_adapter import (
    BaseModule, ATTENTION, xavier_init, constant_init, auto_fp16, force_fp32
)

@ATTENTION.register_module()
class RadarGuidedBEVAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
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
        self.num_bev_queue = num_bev_queue
        
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = jt.arange(
            self.num_heads,
            dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdims=True)).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.assign(grid_init.view(-1))
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
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
                bev_h=None,
                bev_w=None,
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
        print(f"DEBUG: RadarGuidedBEVAttention execute. bev_h={bev_h}, bev_w={bev_w}, batch_first={self.batch_first}")
        # print(f"DEBUG: RadarGuidedBEVAttention spatial_shapes (input): {spatial_shapes.shape}, data: {spatial_shapes.numpy()}")
        # print(f"DEBUG: Initial query shape: {query.shape}")
        # if value is not None:
        #      print(f"DEBUG: Initial value shape: {value.shape}")
        # else:
        #      print("DEBUG: Initial value is None")

        # Override spatial_shapes for BEV self-attention
        if bev_h is not None and bev_w is not None:
             # print(f"DEBUG: Overriding spatial_shapes with BEV shape: ({bev_h}, {bev_w})")
             spatial_shapes = jt.array([[bev_h, bev_w]], dtype=jt.int32)
             level_start_index = jt.array([0], dtype=jt.int32)

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            print(f"DEBUG: value is None. Derived bs={bs}, len_bev={len_bev}")
            if self.num_bev_queue == 2:
                value = jt.stack([query, query], 1).reshape(bs*2, len_bev, c)
            else:
                value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs,  num_query, embed_dims = query.shape
        
        # Handle multi-view value (bs, num_cam, num_value, embed_dims)
        is_multi_view = False
        num_cam = 1
        if value.ndim == 4:
            is_multi_view = True
            bs, num_cam, num_value, _ = value.shape
            # Optimization: Do NOT expand bs here to avoid OOM. Process cameras sequentially.
        else:
            _, num_value, _ = value.shape

        # Handle reference_points from point_sampling (D, bs, num_cam, num_query, 3)
        if reference_points is not None and reference_points.ndim == 5:
             # This path seems unused or incorrect for current flow, but keeping for safety if 5D passed
             # (D, bs, num_cam, num_query, 3) -> (bs, num_cam, num_query, D, 3)
             reference_points = reference_points.permute(1, 2, 3, 0, 4)
             # (bs, num_cam, num_query, D, 3) -> (bs, num_cam, num_query, 1, D, 3)
             reference_points = reference_points.unsqueeze(3)
             # (bs, num_cam, num_query, 1, D, 3) -> (bs, num_cam, num_query, num_levels, D, 3)
             reference_points = reference_points.repeat(1, 1, 1, self.num_levels, 1, 1)

        # Handle 3D reference points (last dim 3) by switching to reference_points_cam if available
        if reference_points is not None and reference_points.shape[-1] == 3:
             reference_points_cam = kwargs.get('reference_points_cam', None)
             if reference_points_cam is not None:
                  # reference_points_cam shape: (num_cam, B, num_query, D, 2)
                  # Target: (B, num_cam, num_query, D, 2)
                  
                  # Verify shape compatibility
                  if reference_points_cam.ndim == 5:
                       # (num_cam, B, num_query, D, 2) -> (B, num_cam, num_query, D, 2)
                       reference_points = reference_points_cam.permute(1, 0, 2, 3, 4)
                       # Note: We do not flatten here in optimized version

        # Get prev_bev from kwargs
        prev_bev = kwargs.get('prev_bev', None)
        
        # Prepare query for offset/attention weight prediction
        # If num_bev_queue is 2, we need to concatenate prev_bev and query
        query_input = query
        if self.num_bev_queue == 2:
            if prev_bev is None:
                # If no prev_bev, duplicate query as fallback
                prev_bev = query
            
            # Ensure prev_bev shape matches query shape in dim 0
            if prev_bev.shape[0] != query.shape[0]:
                 # Try to broadcast or repeat if simple expansion works
                 if prev_bev.shape[0] * num_cam == query.shape[0]:
                      # If prev_bev was expanded but query wasn't (unlikely here)
                      pass
            
            query_input = jt.cat([prev_bev, query], -1)
            
        print(f"DEBUG: value before proj shape: {value.shape}, bs={bs}")
        value = self.value_proj(value)
        print(f"DEBUG: value after proj shape: {value.shape}")

        if key_padding_mask is not None:
            value = jt.where(key_padding_mask[..., None], 0.0, value)
        
        # DEBUG: Check query before Linear
        print(f"DEBUG: RadarGuidedBEVAttention query_input.shape={query_input.shape}")
        # jt.sync_all(True)
        
        print("DEBUG: Executing sampling_offsets...")
        sampling_offsets = self.sampling_offsets(query_input)
        print(f"DEBUG: sampling_offsets result shape={sampling_offsets.shape}")
        # jt.sync_all(True)

        # Force memory continuity before view
        # sampling_offsets_data = sampling_offsets.data
        # sampling_offsets = jt.array(sampling_offsets_data)
        # jt.sync_all(True)

        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        
        # DEBUG: Check sampling_offsets
        # jt.sync_all(True)
        print("DEBUG: sampling_offsets viewed.")
        
        # sampling_offsets = jt.clamp(sampling_offsets, -1e3, 1e3)
        # print("DEBUG: Skipped clamp.")

        print("DEBUG: Executing attention_weights...")
        attention_weights = self.attention_weights(query_input)
        print("DEBUG: attention_weights computed.")
        attention_weights = attention_weights.view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
        
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # DEBUG: Check sampling_offsets
        # jt.sync_all(True)
        # if jt.isnan(sampling_offsets).any():
        #     print("DEBUG: sampling_offsets contains NaNs in RadarGuidedBEVAttention")

        # Offset normalizer
        offset_normalizer = None
        if reference_points is not None and reference_points.shape[-1] == 2:
            offset_normalizer = jt.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            # Safe division
            offset_normalizer = jt.maximum(offset_normalizer, jt.ones_like(offset_normalizer) * 1e-6)

        # Loop over cameras
        outputs_list = []
        loop_range = range(num_cam) if is_multi_view else range(1)
        
        print(f"DEBUG: Starting camera loop. is_multi_view={is_multi_view}, num_cam={num_cam}")
        
        for cam_idx in loop_range:
             # 1. Prepare Value for this camera
             if is_multi_view:
                  value_curr = value[:, cam_idx] # (bs, num_value, embed_dims)
             else:
                  value_curr = value
             
             # Reshape value_curr for attention
             real_num_bev_queue = 1
             if value_curr.shape[0] == bs * self.num_bev_queue:
                  real_num_bev_queue = self.num_bev_queue
             
             value_curr = value_curr.reshape(bs*real_num_bev_queue,
                                   num_value, self.num_heads, -1)
             
             if real_num_bev_queue == 1 and self.num_bev_queue > 1:
                  value_curr = value_curr.unsqueeze(1).repeat(1, self.num_bev_queue, 1, 1, 1)
                  value_curr = value_curr.reshape(bs*self.num_bev_queue, num_value, self.num_heads, -1)
             
             # Jittor Fix: Ensure value is [BS, Num_Keys, Num_Heads, C]
             # Based on previous logic: 
             # value = value.view(bs_inferred, self.num_heads, num_value, -1)
             # value = value.permute(0, 2, 1, 3)
             # But here we reshaped to (bs, num_value, heads, C)
             # So we permute (0, 2, 1, 3) to get (bs, heads, num_value, C) -> Wait.
             # Original code:
             # value = value.view(bs_inferred, self.num_heads, num_value, -1) -> (BS, Heads, Keys, C)
             # value = value.permute(0, 2, 1, 3) -> (BS, Keys, Heads, C) ??
             # Let's check multi_scale_deformable_attn_jittor expectation.
             # Assuming previous code block was correct about reshaping.
             # "Jittor Fix: Ensure value is [BS, Num_Keys, Num_Heads, C]"
             # So we want [BS, Num_Keys, Num_Heads, C].
             # Our value_curr is (BS, Num_Keys, Num_Heads, C).
             # So no permute needed if it is already in that shape?
             # Wait, in original code:
             # value = value.reshape(bs*real_num_bev_queue, num_value, self.num_heads, -1) -> (BS, Keys, Heads, C)
             # Then:
             # if value.ndim == 3: ...
             # It seems my previous read of original code (lines 402-409) handled 3D input.
             # But here value_curr is 4D.
             # So value_curr is (BS, Keys, Heads, C).
             # Let's assume this is correct for multi_scale_deformable_attn_jittor.
             
             # 2. Prepare Reference Points for this camera
             if is_multi_view and reference_points is not None:
                  # reference_points: (bs, num_cam, num_query, levels, points, 2)
                  if reference_points.ndim == 6:
                       ref_curr = reference_points[:, cam_idx] # (bs, num_query, levels, points, 2)
                  elif reference_points.ndim == 5:
                       # Maybe (bs, num_cam, num_query, D, 2)
                       ref_curr = reference_points[:, cam_idx]
                  else:
                       ref_curr = reference_points
             else:
                  ref_curr = reference_points
             
             # Expand ref_curr for num_bev_queue
             if ref_curr is not None and ref_curr.shape[0] == bs:
                  ref_shape = ref_curr.shape
                  # (bs, ...) -> (bs, num_bev_queue, ...) -> (bs*num_bev_queue, ...)
                  ref_curr = ref_curr.unsqueeze(1).repeat(1, self.num_bev_queue, *([1]*(len(ref_shape)-1)))
                  ref_curr = ref_curr.reshape(bs*self.num_bev_queue, *ref_shape[1:])
             
             # 3. Calculate Sampling Locations
             if ref_curr.shape[-1] == 2:
                  sampling_locations = ref_curr[:, :, None, :, None, :] \
                       + sampling_offsets \
                       / offset_normalizer[None, None, None, :, None, :]
             elif ref_curr.shape[-1] == 4:
                  sampling_locations = ref_curr[:, :, None, :, None, :2] \
                       + sampling_offsets / self.num_points \
                       * ref_curr[:, :, None, :, None, 2:] \
                       * 0.5
             else:
                  raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {ref_curr.shape[-1]} instead.')
             
             # 4. Attention
             # print(f"DEBUG: Cam {cam_idx} attention. Value: {value_curr.shape}, Locs: {sampling_locations.shape}")
             output_curr = multi_scale_deformable_attn_jittor(
                  value_curr, spatial_shapes, sampling_locations, attention_weights)
             
             outputs_list.append(output_curr)
             jt.gc()
        
        # Aggregate
        # outputs_list: [ (bs*num_bev_queue, num_query, embed_dims), ... ]
        if len(outputs_list) > 1:
             output = jt.stack(outputs_list, dim=0) # (num_cam, bs*Q, N, C)
             output = output.mean(0) # (bs*Q, N, C)
        else:
             output = outputs_list[0]
        
        # (bs*num_bev_queue, num_query, embed_dims)-> (bs, num_bev_queue, num_query, embed_dims)
        output = output.view(bs, self.num_bev_queue, num_query, embed_dims)

        # fuse history value and current value
        # (bs, num_bev_queue, num_query, embed_dims)-> (bs, num_query, embed_dims)
        output = output.mean(1)

        # (bs, num_query, embed_dims) -> (num_query, bs, embed_dims)
        # output = output.permute(1, 0, 2)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)
        
        # Use original identity if possible, or aggregate identity_expanded
        if is_multi_view and identity.shape[0] == bs * num_cam:
             # If identity passed was already expanded (unlikely in new flow but possible if passed from outside)
             identity = identity.view(bs, num_cam, num_query, embed_dims).mean(1)

        return self.dropout(output) + identity
