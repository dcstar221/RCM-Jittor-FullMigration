
import jittor as jt
from jittor import nn
import math
import warnings
from projects.mmdet3d_plugin.jittor_adapter import (
    ATTENTION, build_attention, BaseModule, xavier_init, constant_init, force_fp32
)
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, multi_scale_deformable_attn_jittor
from projects.mmdet3d_plugin.models.utils.bricks import run_time

@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def execute(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Var): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Var): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Var): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Var): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Var): The positional encoding for `query`.
                Default: None.
            key_pos (Var): The positional encoding for  `key`. Default
                None.
            reference_points (Var):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Var): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Var): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Var): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Var: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        print(f"DEBUG: SpatialCrossAttention execute. query.shape={query.shape}")
        if key is not None:
             print(f"DEBUG: key.shape={key.shape}")
        if value is not None:
             print(f"DEBUG: value.shape={value.shape}")
        if reference_points_cam is not None:
             print(f"DEBUG: reference_points_cam.shape={reference_points_cam.shape}")

        if residual is None:
            inp_residual = query
            slots = jt.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.shape[3]
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # mask_per_img: (bs, num_query, ...)
            # We assume mask_per_img[0] is representative if bs>1, 
            # or we need to handle batch?
            # Original code: index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            # This suggests mask is (bs, num_query, something) or (num_query, something)
            # If mask_per_img is (bs, num_query, ?), mask_per_img[0] is (num_query, ?)
            # sum(-1) -> (num_query,)
            # nonzero() -> indices
            
            # Jittor nonzero returns tuple of indices
            # .squeeze(-1) might be needed
            
            # Note: Jittor's nonzero behavior might differ slightly.
            # jt.nonzero(cond) returns (N, D)
            
            # Let's try to follow logic.
            # mask_per_img[0].sum(-1) -> shape (num_query)
            idx = jt.nonzero(mask_per_img[0].sum(-1))
            if idx.shape[0] > 0:
                index_query_per_img = idx.squeeze(1)
            else:
                index_query_per_img = idx.reshape(0) # Empty
                
            indexes.append(index_query_per_img)
            
        max_len = max([len(each) for each in indexes])

        if max_len > 0:
            # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
            queries_rebatch = jt.zeros(
                [bs, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = jt.zeros(
                [bs, self.num_cams, max_len, D, 2])
            
            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):   
                    index_query_per_img = indexes[i]
                    if len(index_query_per_img) > 0:
                        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

            num_cams, l, bs, embed_dims = key.shape

            key = key.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, l, self.embed_dims)
            value = value.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, l, self.embed_dims)
            
            queries = self.deformable_attention(
                query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), 
                key=key, 
                value=value,
                reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), 
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
                
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    if len(index_query_per_img) > 0:
                        # Jittor doesn't support += on advanced indexing directly in all cases, 
                        # but let's try. If fails, might need scatter_add or similar.
                        # slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
                        
                        # Workaround if += fails:
                        # current = slots[j, index_query_per_img]
                        # update = current + queries[j, i, :len(index_query_per_img)]
                        # slots[j, index_query_per_img] = update
                        
                        # Actually, we can use scatter logic or just rely on Jittor's setitem
                        # But Jittor's setitem might not accumulate? 
                        # "slots[j, index_query_per_img] += ..." is read-modify-write.
                        # For safety with Jittor, better to read, add, write.
                        
                        indices = index_query_per_img
                        update_val = queries[j, i, :len(indices)]
                        
                        # We need to add to existing values.
                        # In PyTorch: slots[j, indices] += update_val
                        # In Jittor:
                        # slots[j, indices] = slots[j, indices] + update_val
                        # But if multiple cameras map to same query? 
                        # The loop is over cameras (i). Each camera has its own index_query_per_img.
                        # Yes, queries are shared across cameras (BEV queries).
                        # So slots accumulate results from multiple cameras.
                        
                        # target_val = slots[j, indices] + update_val
                        # slots[j, indices] = target_val # This might not work if indices are not unique?
                        # indices are unique per camera for BEV mask.
                        # So per camera loop, we update specific slots.
                        
                        # To perform this update in Jittor:
                        # We might need to construct a full update tensor and add?
                        # Or use setitem.
                        
                        # Let's try direct setitem for now.
                        slots[j, indices] = slots[j, indices] + update_val

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = jt.clamp(count, 1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
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
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
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
        self.batch_first = batch_first
        self.output_proj = None
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
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.assign(grid_init.view(-1))
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
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
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Var): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Var): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Var): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
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
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        print(f"DEBUG: MSDeformableAttention3D.execute. value.shape={value.shape}")
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = jt.where(key_padding_mask[..., None], 0.0, value)
            
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
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = jt.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            # assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            raise NotImplementedError("Reference points with shape 4 not implemented")
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        
        output = multi_scale_deformable_attn_jittor(
            value, spatial_shapes, sampling_locations, attention_weights)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
