
import jittor as jt
from jittor import nn
from projects.mmdet3d_plugin.jittor_adapter import ATTENTION, BaseModule

@ATTENTION.register_module()
class MultiheadAttention(BaseModule):
    """Multi-headed attention.
    See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self, embed_dims, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg)
        self.embed_dim = embed_dims
        self.kdim = kdim if kdim is not None else embed_dims
        self.vdim = vdim if vdim is not None else embed_dims
        self._qkv_same_embed_dim = self.kdim == embed_dims and self.vdim == embed_dims

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(jt.empty((embed_dims, embed_dims)))
            self.k_proj_weight = nn.Parameter(jt.empty((embed_dims, self.kdim)))
            self.v_proj_weight = nn.Parameter(jt.empty((embed_dims, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(jt.empty((3 * embed_dims, embed_dims)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(jt.empty((3 * embed_dims)))
            self.out_proj = nn.Linear(embed_dims, embed_dims, bias=True)
        else:
            self.register_parameter('in_proj_bias', None)
            self.out_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.add_bias_kv = add_bias_kv
        if add_bias_kv:
            self.bias_k = nn.Parameter(jt.empty((1, 1, embed_dims)))
            self.bias_v = nn.Parameter(jt.empty((1, 1, embed_dims)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def execute(self, query, key=None, value=None, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        """
        Args:
            query, key, value: map to standard PyTorch forward arguments.
            If key/value are None, self-attention is assumed (key=value=query).
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        if self.batch_first:
            # If batch_first, inputs are (B, L, E)
            # We reshape to (L, B, E) for processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            tgt_len, bsz, embed_dim = query.shape
            src_len = key.shape[0]

        if self._qkv_same_embed_dim:
            # Standard self-attention
            # (L, B, E) -> (L, B, 3*E)
            qkv = nn.linear(query, self.in_proj_weight, self.in_proj_bias)
            # Split
            qkv = qkv.view(tgt_len, bsz, 3, embed_dim)
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]
        else:
            # Not supported by weights, but implemented for completeness
            q = nn.linear(query, self.q_proj_weight, self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
            k = nn.linear(key, self.k_proj_weight, self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
            v = nn.linear(value, self.v_proj_weight, self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)

        # Scaling
        q = q * (self.head_dim ** -0.5)

        # Reshape for multi-head attention: (L, B, E) -> (L, B, H, D) -> (B, H, L, D)
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Attention weights: (B*H, L, S)
        attn_output_weights = jt.bmm(q, k.transpose(1, 2))
        
        if attn_mask is not None:
             attn_output_weights += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, S)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = nn.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.dropout(attn_output_weights, p=self.dropout)

        attn_output = jt.bmm(attn_output_weights, v)
        
        # Reshape back: (B*H, L, D) -> (L, B*H, D) -> (L, B, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            
        if need_weights:
            # Average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None
