
import jittor as jt
from jittor import nn

def multi_scale_deformable_attn_jittor(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    value: (bs, num_keys, num_heads, embed_dims//num_heads)
    value_spatial_shapes: (num_levels, 2)
    sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
    attention_weights: (bs, num_queries, num_heads, num_levels, num_points)
    """
    bs, num_keys, num_heads, c = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    
    print(f"DEBUG: multi_scale_deformable_attn_jittor value.shape={value.shape}")
    
    # Clone inputs to ensure contiguous memory
    value = value.clone()
    sampling_locations = sampling_locations.clone()
    
    print(f"DEBUG: sampling_locations.shape={sampling_locations.shape}")
    # Check sampling_locations validity
    # if jt.isnan(sampling_locations).any():
    #     print("DEBUG: ERROR: sampling_locations contains NaNs!")
    # print(f"DEBUG: sampling_locations min={sampling_locations.min().item()}, max={sampling_locations.max().item()}")
    
    if isinstance(value_spatial_shapes, jt.Var):
         print(f"DEBUG: value_spatial_shapes shape={value_spatial_shapes.shape}")
         print(f"DEBUG: value_spatial_shapes data={value_spatial_shapes.data}")
    else:
         print(f"DEBUG: value_spatial_shapes={value_spatial_shapes}")

    # Calculate split sections from spatial shapes
    if isinstance(value_spatial_shapes, jt.Var):
        spatial_shapes = value_spatial_shapes.data.tolist()
    else:
        spatial_shapes = value_spatial_shapes
        
    split_sections = [int(h * w) for h, w in spatial_shapes]
    
    # Check total length
    total_len = sum(split_sections)
    if total_len != num_keys:
        print(f"DEBUG: Mismatch! num_keys={num_keys}, sum(spatial_shapes)={total_len}")
        # If mismatch, it might be padding or other issues.
        # But here we assume it matches for now.
    
    start = 0
    value_list = []
    
    # Force sync before loop
    jt.sync_all(True)
    
    for length in split_sections:
        # value_list.append(value[:, start:start+length, :, :])
        # Use execute to ensure slice happens?
        v_slice = value[:, start:start+length, :, :]
        value_list.append(v_slice)
        start += length
    
    # [0, 1] -> [-1, 1]
    sampling_grids = 2 * sampling_locations - 1 
    
    output = []
    for level, (h, w) in enumerate(spatial_shapes):
        # value_l: (bs, H*W, num_heads, c) -> (bs, num_heads, c, H, W)
        H, W = int(h), int(w)
        
        # Debug slice shape
        v_slice = value_list[level]
        # Force clone to ensure contiguous memory before reshape
        v_slice = v_slice.clone()
        print(f"DEBUG: Level {level}, v_slice.shape={v_slice.shape}, H={H}, W={W}")
        
        if H * W == 0:
            print(f"DEBUG: Level {level} is empty (H={H}, W={W}), skipping")
            # Create a zero tensor of expected output shape
            # expected: (bs*num_heads, c, num_queries, num_points)
            sample_l = jt.zeros((bs*num_heads, c, num_queries, num_points))
            output.append(sample_l)
            continue
            
        print("DEBUG: Executing value_l transpose...")
        v_transposed = v_slice.transpose(1, 2).clone()
        jt.sync_all(True)
        print(f"DEBUG: v_transposed shape={v_transposed.shape}")

        # Force memory continuity
        v_transposed_data = v_transposed.data
        v_transposed = jt.array(v_transposed_data)
        jt.sync_all(True)
        
        print(f"DEBUG: Executing value_l reshape to ({bs*num_heads}, {c}, {H}, {W})...")
        value_l = v_transposed.reshape(bs*num_heads, c, H, W)
        jt.sync_all(True)
        print(f"DEBUG: value_l reshaped={value_l.shape}")
        
        # grid: (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        print(f"DEBUG: Executing grid_l slice/transpose/reshape...")
        grid_slice = sampling_grids[:, :, :, level, :, :]
        print(f"DEBUG: grid_slice shape={grid_slice.shape}")
        jt.sync_all(True)
        
        grid_transposed = grid_slice.transpose(1, 2).clone()
        print(f"DEBUG: grid_transposed shape={grid_transposed.shape}")
        jt.sync_all(True)

        # Force memory continuity by creating a new array from data
        # This is expensive but guarantees memory layout
        grid_transposed_data = grid_transposed.data
        grid_transposed = jt.array(grid_transposed_data)
        jt.sync_all(True)

        grid_l = grid_transposed.reshape(bs*num_heads, num_queries, num_points, 2)
        print(f"DEBUG: grid_l shape={grid_l.shape}")
        jt.sync_all(True)
        
        # Clamp grid just in case
        grid_l = jt.clamp(grid_l, -1.0, 1.0)

        # Force sync and clone before grid_sample
        value_l = value_l.clone()
        grid_l = grid_l.clone()
        jt.sync_all(True)
        
        # sample_l: (bs*num_heads, c, num_queries, num_points)
        print(f"DEBUG: grid_sample call: value_l.shape={value_l.shape}, grid_l.shape={grid_l.shape}")
        
        # Check grid range
        # print(f"DEBUG: grid_l min={grid_l.min().item()}, max={grid_l.max().item()}")
        
        # Jittor grid_sample arguments check
        try:
            # Sync before critical op
            jt.sync_all(True)
            sample_l = nn.grid_sample(value_l, grid_l, align_corners=False, padding_mode='zeros')
            # Force execution
            jt.sync_all(True)
        except Exception as e:
            print(f"DEBUG: grid_sample failed: {e}")
            raise e
        
        print("DEBUG: grid_sample success")

        
        # weights: (bs, num_queries, num_heads, num_points) -> (bs*num_heads, 1, num_queries, num_points)
        weights_l = attention_weights[:, :, :, level, :].transpose(1, 2).reshape(bs*num_heads, 1, num_queries, num_points)
        
        output.append(sample_l * weights_l)
    
    # Sum over levels
    # output = sum(output) # (bs*num_heads, c, num_queries, num_points)
    # Jittor sum of list?
    if len(output) > 1:
        output_sum = output[0]
        for i in range(1, len(output)):
            output_sum = output_sum + output[i]
    else:
        output_sum = output[0]
    
    # Sum over points
    output_sum = output_sum.sum(-1) # (bs*num_heads, c, num_queries)
    
    # Reshape back: (bs*num_heads, c, num_queries) -> (bs, num_heads, c, num_queries) -> (bs, num_queries, num_heads, c)
    output_final = output_sum.reshape(bs, num_heads, c, num_queries).permute(0, 3, 1, 2)
    output_final = output_final.reshape(bs, num_queries, num_heads*c)
    
    return output_final

class MultiScaleDeformableAttnFunction_fp32:
    @staticmethod
    def apply(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        return multi_scale_deformable_attn_jittor(value, value_spatial_shapes, sampling_locations, attention_weights)
