
import jittor as jt
from jittor import nn
import numpy as np

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = jt.zeros((B, npoint), dtype='int32')
    distance = jt.ones((B, N)) * 1e10
    farthest = jt.randint(0, N, (B,), dtype='int32')
    batch_indices = jt.arange(B, dtype='int32')
    
    for i in range(npoint):
        centroids[:, i] = farthest
        # xyz[batch_indices, farthest, :]
        # Jittor indexing:
        # We need to gather the farthest point coordinates
        # shape (B, 1, 3)
        
        # Efficient way using gather/indexing
        # flat_indices = batch_indices * N + farthest
        # centroid = xyz.reshape(B*N, 3)[flat_indices].view(B, 1, 3)
        
        # Or simply:
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        dist = jt.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = jt.where(mask, dist, distance)
        
        # farthest = jt.argmax(distance, -1)[0] # Jittor argmax returns (idx, val)
        farthest, _ = jt.argmax(distance, -1)
        
    return centroids

def ball_query(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, M, 3]
    Return:
        group_idx: grouped points index, [B, M, nsample]
    """
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    
    # sq_dist: [B, M, N]
    # new_xyz: [B, M, 1, 3]
    # xyz: [B, 1, N, 3]
    sq_dist = jt.sum((new_xyz.view(B, M, 1, 3) - xyz.view(B, 1, N, 3)) ** 2, -1)
    
    # Find k nearest neighbors to ensure we have nsample points
    # We use topk on negative distance to find smallest distances
    # vals, idx = jt.topk(-sq_dist, nsample, dim=-1) # (B, M, nsample)
    
    # If N < nsample, we take all N and repeat
    if N < nsample:
        # Not handling this case for now, assuming N >= nsample
        pass
        
    vals, idx = jt.topk(-sq_dist, k=nsample, dim=-1)
    
    # Mask out points outside radius
    mask = (-vals) > (radius ** 2)
    
    # For points outside radius, we should repeat the first point (usually)
    # idx: [B, M, nsample]
    # group_first: [B, M, 1]
    group_first = idx[:, :, 0:1].repeat(1, 1, nsample)
    
    # Apply mask: where mask is True (dist > radius^2), replace with first point
    # Wait, mask is True if dist > radius^2?
    # vals is negative dist^2. -vals is dist^2.
    # So if (-vals) > radius^2, it is outside.
    
    idx = jt.where(mask, group_first, idx)
    
    return idx

class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz

    def execute(self, xyz, new_xyz, features=None):
        """
        xyz: [B, N, 3]
        new_xyz: [B, M, 3]
        features: [B, C, N]
        """
        if features is not None:
             print(f"DEBUG: QueryAndGroup features shape: {features.shape}")
             
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # [B, M, nsample]
        B, M, nsample = idx.shape
        B, N, C = xyz.shape
        
        # Gather xyz
        # xyz_trans: [B, 3, N]
        xyz_trans = xyz.permute(0, 2, 1)
        # grouped_xyz: [B, 3, M, nsample]
        # We need to gather from axis 2 (N) using idx
        
        # Flatten for gather
        # idx_flat: [B, M*nsample]
        # xyz_flat: [B, 3, N]
        
        # Jittor gather: jt.gather(input, dim, index)
        # index must have same dims as input? No.
        # "The shape of index must be the same as input except for the dimension dim"
        # This is restrictive.
        
        # Use fancy indexing with flattened batch
        # idx: [B, M, nsample] -> offset by batch
        batch_offset = jt.arange(B).view(B, 1, 1) * N
        idx_flat = (idx + batch_offset).view(-1) # [B*M*nsample]
        
        xyz_flat = xyz.view(B*N, 3) # [B*N, 3]
        grouped_xyz = xyz_flat[idx_flat, :].view(B, M, nsample, 3) # [B, M, nsample, 3]
        
        # new_xyz: [B, M, 1, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, M, 1, 3)
        
        if features is not None:
            # features: [B, C, N] -> [B, N, C]
            features_trans = features.permute(0, 2, 1) # [B, N, C]
            features_flat = features_trans.reshape(B*N, -1) # [B*N, C]
            # print(f"DEBUG: features_flat shape: {features_flat.shape}")
            grouped_features = features_flat[idx_flat, :].view(B, M, nsample, -1) # [B, M, nsample, C]
            # print(f"DEBUG: grouped_features before permute: {grouped_features.shape}")
            # [B, C, M, nsample]
            grouped_features = grouped_features.permute(0, 3, 1, 2)
        else:
            grouped_features = None
            
        if self.use_xyz:
            # grouped_xyz_norm: [B, M, nsample, 3] -> [B, 3, M, nsample]
            grouped_xyz_norm_trans = grouped_xyz_norm.permute(0, 3, 1, 2)
            if grouped_features is not None:
                grouped_features = jt.concat([grouped_xyz_norm_trans, grouped_features], dim=1)
            else:
                grouped_features = grouped_xyz_norm_trans
                
        return grouped_features # [B, C+3, M, nsample]

class PointSAModule(nn.Module):
    def __init__(self, num_point, radii, sample_nums, mlp_channels, use_xyz=True, bias=True):
        super(PointSAModule, self).__init__()
        self.num_point = num_point
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        # Handle list inputs (MSG) or single inputs (SSG)
        if isinstance(radii, tuple):
            radii = list(radii)
        elif not isinstance(radii, list):
            radii = [radii]
            
        if isinstance(sample_nums, tuple):
            sample_nums = list(sample_nums)
        elif not isinstance(sample_nums, list):
            sample_nums = [sample_nums]
            
        if isinstance(mlp_channels, tuple):
            mlp_channels = list(mlp_channels)
        elif not isinstance(mlp_channels, list):
            mlp_channels = [mlp_channels]
            
        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            channel_list = mlp_channels[i]
            
            self.groupers.append(QueryAndGroup(radius, sample_num, use_xyz=use_xyz))
            
            mlp = nn.Sequential()
            input_channel = channel_list[0] 
            # Note: channel_list[0] usually expects input feature dim.
            # If use_xyz, QueryAndGroup adds 3 to input dim.
            # But usually mlp_channels config already accounts for it or we adjust it.
            # In MMDetection3D, mlp_channels[0] is modified to +=3 if use_xyz.
            # But here we assume input_channels matches what QueryAndGroup outputs?
            # Let's check InstanceLevelFusion usage.
            
            for j in range(len(channel_list) - 1):
                in_ch = channel_list[j]
                out_ch = channel_list[j+1]
                if j == 0 and use_xyz:
                     in_ch += 3 # Adjust first layer input
                
                mlp.add_module(f'conv{j}', nn.Conv2d(in_ch, out_ch, 1, bias=bias))
                mlp.add_module(f'bn{j}', nn.BatchNorm2d(out_ch))
                mlp.add_module(f'relu{j}', nn.ReLU())
            self.mlps.append(mlp)

    def execute(self, points_xyz, features=None, new_xyz=None):
        """
        points_xyz: [B, N, 3]
        features: [B, C, N]
        new_xyz: [B, M, 3]
        """
        if new_xyz is None:
            # If new_xyz not provided, use FPS to sample
            # This is not used in InstanceLevelFusion but good to have
            idx = furthest_point_sample(points_xyz, self.num_point)
            # Gather new_xyz
            B, N, _ = points_xyz.shape
            batch_indices = jt.arange(B).view(B, 1)
            # new_xyz = points_xyz[batch_indices, idx, :]
            # Jittor indexing ... simplified:
            flat_idx = idx + batch_indices * N
            new_xyz = points_xyz.view(B*N, 3)[flat_idx.view(-1)].view(B, self.num_point, 3)
            
        new_features_list = []
        for i, grouper in enumerate(self.groupers):
            # grouped_features: [B, C+3, M, nsample]
            grouped_features = grouper(points_xyz, new_xyz, features)
            
            # MLP
            # grouped_features = self.mlps[i](grouped_features)
            # Jittor Sequential call
            x = grouped_features
            # print(f"DEBUG: PointSAModule group {i}, input shape {x.shape}")
            for layer_idx, layer in enumerate(self.mlps[i].layers.values()):
                 # print(f"DEBUG: Layer {layer_idx}, type {type(layer)}")
                 if isinstance(layer, nn.Conv2d):
                     if x.shape[1] != layer.in_channels:
                         print(f"DEBUG: Shape Mismatch! x.shape={x.shape}, layer.in_channels={layer.in_channels}")
                 x = layer(x)
            grouped_features = x
            
            # Max Pool
            # [B, out_ch, M, nsample] -> [B, out_ch, M]
            grouped_features = jt.max(grouped_features, dim=-1) # (B, out_ch, M)
            new_features_list.append(grouped_features)
            
        # Concat multi-scale features
        new_features = jt.concat(new_features_list, dim=1)
        
        return new_xyz, new_features

def build_sa_module(num_point, radii, sample_nums, mlp_channels, dilated_group=False, norm_cfg=None, cfg=None, bias=True, **kwargs):
    # Simplified builder
    # Ignores dilated_group, norm_cfg, cfg for now
    return PointSAModule(num_point, radii, sample_nums, mlp_channels, bias=bias)

