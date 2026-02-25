"""
Jittor Version of DynamicPillarVFESimple2D (Voxel/Pillar Feature Encoder)
Ported from PyTorch version.
Key changes:
  - Replace torch_scatter with custom Jittor/Numpy implementation (jt_scatter_mean/max)
  - Replace torch.* with jt.*
  - Register to VOXEL_ENCODERS
"""
import jittor as jt
import jittor.nn as nn
import numpy as np

from projects.mmdet3d_plugin.jittor_adapter import VOXEL_ENCODERS

# -----------------------------------------------------------------------------
# Jittor scatter helper functions (replacing torch_scatter)
# -----------------------------------------------------------------------------
def jt_scatter_mean(src, index, dim=0, dim_size=None):
    """Scatter mean via Jittor reindex_reduce."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
        
    if dim != 0:
        # Move target dim to 0
        src = src.transpose(0, dim)
        
    N = src.shape[0]
    C = src.shape[1] if src.ndim > 1 else 1
    
    # Target shape
    out_shape = [dim_size] + list(src.shape[1:])
    
    # Initialize target with zeros
    target = jt.zeros(out_shape, dtype=src.dtype)
    
    # Use jt.scatter if available (it is in 1.3.8+)
    # scatter(input, dim, index, src, reduce='add')
    out_sum = jt.scatter(target, 0, index, src, reduce='add')

    # Count
    # Create ones with same shape as src's dim 0
    ones = jt.ones([N], dtype=src.dtype)
    # Scatter count
    # For count, we need 1D target of size dim_size
    count_target = jt.zeros([dim_size], dtype=src.dtype)
    out_count = jt.scatter(count_target, 0, index, ones, reduce='add')
    
    # Avoid div by zero
    out_count = jt.maximum(out_count, 1.0)
    
    if src.ndim > 1:
        out_count = out_count.reshape([dim_size] + [1] * (src.ndim - 1))
        
    res = out_sum / out_count
    
    if dim != 0:
        res = res.transpose(0, dim)
        
    return res

def jt_scatter_max(src, index, dim=0, dim_size=None):
    """Scatter max via Jittor scatter."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if dim != 0:
        src = src.transpose(0, dim)

    # Target shape
    out_shape = [dim_size] + list(src.shape[1:])
    
    # Initialize target with zeros (assuming ReLU features, otherwise -inf)
    # Since PFNLayer uses ReLU, zeros is fine.
    target = jt.zeros(out_shape, dtype=src.dtype)
    
    out_max = jt.scatter(target, 0, index, src, reduce='max')

    if dim != 0:
        out_max = out_max.transpose(0, dim)
        
    return out_max, None # argmax not implemented

# -----------------------------------------------------------------------------
# PFNLayerV2
# -----------------------------------------------------------------------------
class PFNLayerV2(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            
        self.relu = nn.ReLU()

    def execute(self, inputs, unq_inv, dim_size=None):
        # inputs: [N, C]
        x = self.linear(inputs)
        if self.use_norm:
            x = self.norm(x)
        x = self.relu(x)
        
        # Max pooling over pillars
        x_max, _ = jt_scatter_max(x, unq_inv, dim=0, dim_size=dim_size)
        
        if self.last_vfe:
            return x_max
        else:
            # Concatenate features: [x, global_max_pooled_repeated]
            # unq_inv maps each point to its pillar index
            x_concatenated = jt.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated

# -----------------------------------------------------------------------------
# DynamicPillarVFESimple2D
# -----------------------------------------------------------------------------
@VOXEL_ENCODERS.register_module()
class DynamicPillarVFESimple2D(nn.Module):
    """
    Dynamic Pillar Voxel Feature Encoder (Jittor Version).
    """

    def __init__(self,
                 num_point_features=4,
                 voxel_size=[0.2, 0.2, 8],
                 grid_size=[512, 512, 1],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 num_filters=[64],
                 with_distance=False,
                 use_absolute_xyz=True,
                 with_cluster_center=True,
                 use_norm=True,
                 legacy=False,
                 **kwargs):
        super().__init__()
        self.use_norm = use_norm
        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz
        self.with_cluster_center = with_cluster_center
        self.legacy = legacy
        
        self.num_point_features = num_point_features
        self.voxel_size = jt.array(np.array(voxel_size, dtype=np.float32))
        self.point_cloud_range = jt.array(np.array(point_cloud_range, dtype=np.float32))
        self.grid_size = jt.array(np.array(grid_size, dtype=np.float32))
        
        # Calculate initial input channels
        self.num_input_features = self.num_point_features
        # if self.use_absolute_xyz:
        #     self.num_input_features += 3
        if self.with_cluster_center:
            self.num_input_features += 3
        if self.with_distance:
            self.num_input_features += 1
            
        # Create PFN layers
        num_filters = [self.num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        # Initialize weights (basic)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def voxelize(self, points_list):
        print("DEBUG: DynamicPillarVFESimple2D voxelize start")
        print(f"DEBUG: points_list len: {len(points_list)}")
        print(f"DEBUG: points[0] shape: {points_list[0].shape}")
        
        voxel_size = self.voxel_size
        pc_range = self.point_cloud_range
        
        points_coords_list = []
        points_features_list = []
        
        for i, points in enumerate(points_list):
            print(f"DEBUG: Processing sample {i}")
            # points is Jittor var, convert to numpy for voxelization logic if needed
            # But here we use Jittor ops if possible, or numpy?
            # The implementation uses numpy() conversion.
            points_np = points.numpy()
            print(f"DEBUG: Sample {i} converted to numpy")
            
            points_valid = points_np
            # Filter points outside range...
            # ...
            
        print("DEBUG: Loop finished")
        
        print("DEBUG: Getting voxel_size numpy")
        voxel_size = self.voxel_size.numpy()
        print("DEBUG: Getting pc_range numpy")
        pc_range = self.point_cloud_range.numpy()
        print("DEBUG: Getting grid_size numpy")
        grid_size = self.grid_size.numpy()
        print("DEBUG: Configs loaded")
        
        features_list = []
        coors_list = []
        
        for i, points in enumerate(points_list):
            print(f"DEBUG: Processing real loop sample {i}")
            if isinstance(points, jt.Var):
                points_np = points.numpy()
            else:
                points_np = points
            
            print(f"DEBUG: Sample {i} converted/checked")
            
            # Filter points outside range
            mask = (points_np[:, 0] >= pc_range[0]) & (points_np[:, 0] < pc_range[3]) & \
                   (points_np[:, 1] >= pc_range[1]) & (points_np[:, 1] < pc_range[4]) & \
                   (points_np[:, 2] >= pc_range[2]) & (points_np[:, 2] < pc_range[5])
            points_valid = points_np[mask]
            
            print(f"DEBUG: Sample {i} filtered, valid: {points_valid.shape}")
            
            if points_valid.shape[0] == 0:
                print(f"DEBUG: Sample {i} is empty")
                continue
            
            # Calculate voxel indices
            # x_idx = floor((x - x_min) / x_voxel_size)
            x_idx = np.floor((points_valid[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
            y_idx = np.floor((points_valid[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)
            z_idx = np.floor((points_valid[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32)
            
            print(f"DEBUG: Sample {i} indices calculated")
            
            # Clip to grid size (just in case, though range check should handle it)
            x_idx = np.clip(x_idx, 0, int(grid_size[0]) - 1)
            y_idx = np.clip(y_idx, 0, int(grid_size[1]) - 1)
            z_idx = np.clip(z_idx, 0, int(grid_size[2]) - 1)
            
            # Create coors: (batch_idx, z, y, x)
            batch_idx = np.full((points_valid.shape[0], 1), i, dtype=np.int32)
            coors_i = np.stack([z_idx, y_idx, x_idx], axis=1)
            coors_i = np.concatenate([batch_idx, coors_i], axis=1)
            
            print(f"DEBUG: Sample {i} coors created")
            
            features_list.append(points_valid)
            coors_list.append(coors_i)
            
        if len(features_list) == 0:
            print("DEBUG: No features found")
            return jt.zeros((0, self.num_point_features)), jt.zeros((0, 4))
            
        print("DEBUG: Concatenating features and coors")
        features = np.concatenate(features_list, axis=0)
        coors = np.concatenate(coors_list, axis=0)
        
        print("DEBUG: Converting features to Jittor array (keeping coors as numpy)")
        return jt.array(features), coors

    def execute(self, features, coors=None):
        """
        Args:
            features: [N, num_point_features] OR list of [N, C] points
            coors: [N, 4] (batch_idx, z, y, x) OR None (if features is list of points)
        Returns:
            pillar_features: [M, C]
            coords: [M, 4] (batch_idx, z, y, x)
        """
        # Handle dynamic voxelization if input is list of points
        if isinstance(features, list) or (isinstance(features, (tuple)) and len(features) > 0 and (isinstance(features[0], jt.Var) or isinstance(features[0], np.ndarray))):
             # Assume features is points_list
             print("DEBUG: Executing voxelize...")
             features, coors = self.voxelize(features)
             print("DEBUG: Voxelize done")
             
        features_ls = [features]
        
        # Find distance of x, y, z from cluster center
        print("DEBUG: Calculating unique voxels")
        
        # Use numpy for unique to be safe
        if isinstance(coors, jt.Var):
            print("DEBUG: Converting coors to numpy")
            coors_np = coors.numpy().astype(np.int32)
            print("DEBUG: Coors converted")
        else:
            print("DEBUG: Coors is already numpy/list")
            coors_np = coors.astype(np.int32) if isinstance(coors, np.ndarray) else np.array(coors, dtype=np.int32)
        
        # coors is [N, 4]
        # We need unique rows.
        # np.unique with axis=0 returns unique rows.
        # return_inverse=True gives us the mapping from original points to unique voxels.
        print("DEBUG: Calling np.unique")
        unique_coors, unq_inv = np.unique(coors_np, axis=0, return_inverse=True)
        print("DEBUG: np.unique done")
        
        dim_size = unique_coors.shape[0]

        print("DEBUG: Creating unq_inv jt array")
        unq_inv = jt.array(unq_inv.astype(np.int32)) # [N]
        # Ensure unique_coors is contiguous and safe
        unique_coors_safe = unique_coors.astype(np.int32).copy()
        # Use tolist to avoid memory issues
        print("DEBUG: Creating unique_coors_jt array")
        unique_coors_jt = jt.array(unique_coors_safe.tolist())
        
        # Calculate geometric features
        if self.use_absolute_xyz:
            # features[:, :3] is x, y, z
            pass # already included in features_ls
            
        if self.with_cluster_center:
            print("DEBUG: Calculating cluster center")
            points_mean = jt_scatter_mean(features[:, :3], unq_inv, dim=0, dim_size=dim_size) # [M, 3]
            f_cluster = features[:, :3] - points_mean[unq_inv, :]
            features_ls.append(f_cluster)
            
        if self.with_distance:
            points_dist = jt.norm(features[:, :3], dim=1, keepdim=True)
            features_ls.append(points_dist)
            
        # Combine all features
        features = jt.cat(features_ls, dim=-1)
        
        # Forward through PFN layers
        print("DEBUG: Forwarding PFN layers")
        for i, pfn in enumerate(self.pfn_layers):
            features = pfn(features, unq_inv, dim_size=dim_size)
            
        print("DEBUG: PFN done")
        return features, unique_coors_jt
