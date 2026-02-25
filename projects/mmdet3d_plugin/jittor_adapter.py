
import jittor as jt
import numpy as np
import os
import sys
from collections import OrderedDict
import jittor.nn as nn

# ==================================================================
# Base Classes and Utils
# ==================================================================

class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def force_fp32(apply_to=None):
    def wrapper(func):
        return func
    return wrapper

def auto_fp16(apply_to=None):
    def wrapper(func):
        return func
    return wrapper

# ==================================================================
# Config Implementation (Simplified)
# ==================================================================

class Config(object):
    @staticmethod
    def fromfile(filename):
        return Config(filename)

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict
        self._filename = filename
    
    def __getattr__(self, name):
        return self._cfg_dict.get(name)

    def __getitem__(self, name):
        return self._cfg_dict[name]
    
    def get(self, key, default=None):
        return self._cfg_dict.get(key, default)
    
    def copy(self):
        return Config(self._cfg_dict.copy(), self._filename)
    
    def update(self, other):
        self._cfg_dict.update(other)

# ==================================================================
# Registry Implementation
# ==================================================================

class Registry:
    def __init__(self, name):
        self._module_dict = {}
        self._name = name

    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            self._register_module(module, name, force)
            return module
        
        def _register(cls):
            self._register_module(cls, name, force)
            return cls
        return _register

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
             pass
        self._module_dict[module_name] = module_class

    def get(self, key):
        return self._module_dict.get(key, None)

    @property
    def module_dict(self):
        return self._module_dict

def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    
    if isinstance(cfg, Config):
        cfg = cfg._cfg_dict
    
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
            
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry._name} registry')
    elif isinstance(obj_type, type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    
    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}') from e

# ==================================================================
# Standard Registries
# ==================================================================

DETECTORS = Registry('detector')
BACKBONES = Registry('backbone')
HEADS = Registry('head')
NECKS = Registry('neck')
VOXEL_ENCODERS = Registry('voxel_encoder')
MIDDLE_ENCODERS = Registry('middle_encoder')
FUSION_LAYERS = Registry('fusion_layer')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
MODELS = Registry('model') 
TRANSFORMER = Registry('transformer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer_layer_sequence')
ATTENTION = Registry('attention')
TRANSFORMER_LAYER = Registry('transformer_layer')
FEEDFORWARD_NETWORK = Registry('feedforward_network')
POSITIONAL_ENCODING = Registry('positional_encoding')
BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')
MATCH_COST = Registry('match_cost')

@FEEDFORWARD_NETWORK.register_module()
class FFN(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.0,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = dropout_layer
        self.add_identity = add_identity

    def execute(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out

@ATTENTION.register_module()
class MultiheadAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None,
                 batch_first=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == embed_dims, "embed_dims must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

    def execute(self, 
                query, 
                key=None, 
                value=None, 
                identity=None, 
                query_pos=None, 
                key_pos=None, 
                attn_mask=None, 
                key_padding_mask=None, 
                **kwargs):
        """Forward function for `MultiheadAttention`.
        
        Args:
            query (Tensor): Input query with shape [bs, num_query, embed_dims].
            key (Tensor): Key tensor with shape [bs, num_key, embed_dims].
            value (Tensor): Value tensor with shape [bs, num_key, embed_dims].
            identity (Tensor): The tensor used for addition, with the same shape as `query`. 
                Default None. If None, `query` will be used.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default: None.
            attn_mask (Tensor): ByteTensor mask with shape [num_query, num_key].
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].
            
        Returns:
            Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        if identity is None:
            identity = query
            
        if query_pos is not None:
            if self.batch_first:
                query = query + query_pos
            else:
                # If query_pos is (N, B, E) and query is (N, B, E)
                query = query + query_pos
                
        if key_pos is not None:
            if self.batch_first:
                key = key + key_pos
            else:
                key = key + key_pos

        # Input shape: (L, N, E) if batch_first=False, else (N, L, E)
        if self.batch_first:
            bs, tgt_len, _ = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bs, _ = query.shape
            src_len, _, _ = key.shape
            
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (B, L, N_head, Head_dim) if batch_first=True
        # (L, B, N_head, Head_dim) if batch_first=False
        
        if self.batch_first:
            q = q.view(bs, tgt_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # (B, N_h, L, H_d)
            k = k.view(bs, src_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # (B, N_h, S, H_d)
            v = v.view(bs, src_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # (B, N_h, S, H_d)
        else:
            q = q.view(tgt_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # (B, N_h, L, H_d)
            k = k.view(src_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # (B, N_h, S, H_d)
            v = v.view(src_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # (B, N_h, S, H_d)

        # Attention scores
        attn_output_weights = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scaling # (B, N_h, L, S)
        
        if attn_mask is not None:
            # attn_mask shape: (L, S) or (B*N_h, L, S) or (N_h*B, L, S)??
            # PyTorch: 2D (L, S) or 3D (B*num_heads, L, S)
            # We assume compatible shape or broadcast
            if attn_mask.dtype == jt.bool:
                 attn_output_weights = attn_output_weights.masked_fill(attn_mask, float('-inf'))
            else:
                 attn_output_weights += attn_mask

        if key_padding_mask is not None:
             # key_padding_mask: (B, S)
             # We need to reshape to (B, 1, 1, S) to broadcast
             mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
             attn_output_weights = attn_output_weights.masked_fill(mask, float('-inf'))

        attn_output_weights = nn.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout_layer(attn_output_weights)
        
        attn_output = jt.matmul(attn_output_weights, v) # (B, N_h, L, H_d)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 2, 1, 3).contiguous().view(bs, tgt_len, self.embed_dims)
        else:
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bs, self.embed_dims)
            
        attn_output = self.out_proj(attn_output)
        
        return self.dropout_layer(attn_output) + identity

# ==================================================================
# Builder Helpers
# ==================================================================

@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding(BaseModule):
    def __init__(self, num_feats=128, temperature=10000, normalize=False, scale=2 * np.pi, eps=1e-6, offset=0., init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            self.normalize = normalize
            self.scale = scale
            self.eps = eps
            self.offset = offset
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize

    def execute(self, mask):
        assert mask is not None
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        
        dim_t = jt.arange(self.num_feats, dtype=jt.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = jt.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = jt.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = jt.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_transformer(cfg):
    return build_from_cfg(cfg, TRANSFORMER)

def build_transformer_layer_sequence(cfg):
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE)

def build_attention(cfg):
    return build_from_cfg(cfg, ATTENTION)

def build_feedforward_network(cfg):
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK)

def build_positional_encoding(cfg):
    return build_from_cfg(cfg, POSITIONAL_ENCODING)

def build_transformer_layer(cfg):
    return build_from_cfg(cfg, TRANSFORMER_LAYER)

def build_norm_layer(cfg, num_features, postfix=''):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type == 'LN':
        layer = nn.LayerNorm(num_features, **cfg_)
    elif layer_type == 'BN':
        layer = nn.BatchNorm(num_features, **cfg_)
    elif layer_type == 'GN':
        layer = nn.GroupNorm(num_features, **cfg_)
    else:
        raise NotImplementedError(f'Norm layer {layer_type} is not supported')
    return layer_type, layer

def build_activation_layer(cfg):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    
    # Jittor activations don't support inplace argument
    if 'inplace' in cfg_:
        cfg_.pop('inplace')
        
    if layer_type == 'ReLU':
        layer = nn.ReLU(**cfg_)
    elif layer_type == 'GELU':
        layer = nn.GELU(**cfg_)
    elif layer_type == 'LeakyReLU':
        layer = nn.LeakyReLU(**cfg_)
    else:
        raise NotImplementedError(f'Activation layer {layer_type} is not supported')
    return layer

def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build_from_cfg(cfg, DETECTORS, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)

def build_head(cfg):
    return build_from_cfg(cfg, HEADS)

def build_neck(cfg):
    return build_from_cfg(cfg, NECKS)

def build_voxel_encoder(cfg):
    return build_from_cfg(cfg, VOXEL_ENCODERS)

def build_middle_encoder(cfg):
    return build_from_cfg(cfg, MIDDLE_ENCODERS)

def build_fusion_layer(cfg):
    return build_from_cfg(cfg, FUSION_LAYERS)

def build_bbox_coder(cfg):
    return build_from_cfg(cfg, BBOX_CODERS)

def build_assigner(cfg):
    return build_from_cfg(cfg, BBOX_ASSIGNERS)

def build_sampler(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)

def build_match_cost(cfg):
    return build_from_cfg(cfg, MATCH_COST)
