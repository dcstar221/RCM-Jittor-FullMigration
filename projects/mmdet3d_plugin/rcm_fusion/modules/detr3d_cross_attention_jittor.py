
import numpy as np
import jittor as jt
from jittor import nn
from projects.mmdet3d_plugin.jittor_adapter import xavier_init, constant_init, ATTENTION
import warnings

def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    # Jittor doesn't have direct nan_to_num, implement using where
    x = jt.where(jt.isnan(x), jt.full_like(x, nan), x)
    if posinf is not None:
        x = jt.where(jt.isinf(x) & (x > 0), jt.full_like(x, posinf), x)
    if neginf is not None:
        x = jt.where(jt.isinf(x) & (x < 0), jt.full_like(x, neginf), x)
    return x

@ATTENTION.register_module()
class Detr3DCrossAtten(nn.Module):
    """An attention module used in Detr3d. 
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
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=1,
                 num_cams=6,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 input_dim=64,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pc_range = pc_range

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

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.query_embed = nn.Linear(input_dim, self.embed_dims)
        self.attention_weights = nn.Linear(self.embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
        )
        self.outputs = nn.Linear(self.embed_dims, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def execute(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_metas=None):
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
            key_pos (Var): The positional encoding for `key`. Default
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

        if query_pos is not None:
            query = query + query_pos
        
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        query_embed = self.query_embed(query)
        if residual is None:
            inp_residual = query_embed.permute(1, 0, 2)

        attention_weights = self.attention_weights(query_embed).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, img_metas)
        output = nan_to_num(output)
        mask = nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(reference_points_3d).permute(1, 0, 2)
        return self.norm(self.outputs(self.dropout(output) + inp_residual + pos_feat))


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    # img_metas is a dict, need to extract lidar2img
    # Assume img_metas contains 'lidar2img' which is a numpy array or list
    if isinstance(img_metas, dict):
        lidar2img = img_metas['lidar2img']
    else:
        # If it's a list of dicts (batch), we might need to handle it differently
        # But here img_metas seems to be a single dict for the current batch item?
        # Check usage in instance_level_fusion.py: 
        # sampled_feat = self.Detr3DCrossAtten(..., img_metas=img_metas[i])
        # So img_metas is a single dict.
        lidar2img = img_metas['lidar2img']
        
    lidar2img = np.asarray(lidar2img) # [1, 1, 4, 4] or [N, 4, 4]
    
    # In original code: reference_points.new_tensor(lidar2img)
    # reference_points is a Jittor var.
    lidar2img = jt.array(lidar2img) # (B, N, 4, 4) or (N, 4, 4)
    
    # If lidar2img is (N, 4, 4), add batch dimension
    if lidar2img.ndim == 3:
        lidar2img = lidar2img.unsqueeze(0)
        
    if lidar2img.dtype != reference_points.dtype:
        lidar2img = lidar2img.cast(reference_points.dtype)

    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    # reference_points (B, num_queries, 4)
    
    # torch.cat -> jt.concat
    reference_points = jt.concat((reference_points, jt.ones_like(reference_points[..., :1])), -1) # cam_to_hom
    B, num_query = reference_points.shape[:2]
    num_cam = lidar2img.shape[1] # Assuming lidar2img shape is correct
    
    # view -> view, repeat -> repeat, unsqueeze -> unsqueeze
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    
    # torch.matmul -> jt.matmul
    reference_points_cam = jt.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    
    # torch.maximum -> jt.maximum
    reference_points_cam_div = jt.maximum(
        reference_points_cam[..., 2:3], 
        jt.ones_like(reference_points_cam[..., 2:3])*eps
    )
    
    # Avoid in-place modification if possible or use clone/new var
    # Jittor variables are immutable? No, but operations return new vars.
    # We can do reference_points_cam_xy = ...
    reference_points_cam_xy = reference_points_cam[..., 0:2] / reference_points_cam_div
    
    # img_metas['img_shape'] is likely [[H, W, ...]]
    img_shape = img_metas['img_shape']
    # If img_shape is list of lists, take first one
    if isinstance(img_shape, list) and isinstance(img_shape[0], (list, tuple)):
        H, W = img_shape[0][:2]
    else:
        H, W = img_shape[:2]

    # Normalize
    # reference_points_cam_xy[..., 0] /= W
    # reference_points_cam_xy[..., 1] /= H
    
    x = reference_points_cam_xy[..., 0] / W
    y = reference_points_cam_xy[..., 1] / H
    
    reference_points_cam_norm = jt.stack([x, y], dim=-1)
    
    reference_points_cam_norm = (reference_points_cam_norm - 0.5) * 2
    
    mask = (mask & (reference_points_cam_norm[..., 0:1] > -1.0) 
                 & (reference_points_cam_norm[..., 0:1] < 1.0) 
                 & (reference_points_cam_norm[..., 1:2] > -1.0) 
                 & (reference_points_cam_norm[..., 1:2] < 1.0))
    
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = nan_to_num(mask)
    sampled_feats = []
    
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.shape
        feat = feat.view(B*N, C, H, W).float32()
        reference_points_cam_lvl = reference_points_cam_norm.view(B*N, num_query, 1, 2)
        # F.grid_sample -> nn.grid_sample
        # Jittor grid_sample expects (N, C, H, W) and (N, H_out, W_out, 2)
        sampled_feat = nn.grid_sample(feat, reference_points_cam_lvl, align_corners=False)
        sampled_feats.append(sampled_feat)
    
    sampled_feats = jt.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, N, C, num_query, 1, len(mlvl_feats))
    sampled_feats = sampled_feats.permute(0, 2, 3, 1, 4, 5)
    
    return reference_points_3d, sampled_feats, mask
