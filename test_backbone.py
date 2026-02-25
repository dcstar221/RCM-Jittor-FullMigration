
import jittor as jt
import numpy as np
import sys
import os

# Mock dependencies if needed
# sys.modules['spconv'] = MagicMock()
# sys.modules['spconv.pytorch'] = MagicMock()

from projects.mmdet3d_plugin.models.backbones.spconv_backbone_2d import PillarRes18BackBone8x2

def test_backbone():
    print("Testing PillarRes18BackBone8x2...")
    
    # Config
    voxel_size = [0.2, 0.2, 8]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    # Grid size: [345, 396] (W, H) or similar
    # In config: grid_size = [(69.12 - 0)/0.2, (39.68 - (-39.68))/0.2] = [345.6, 396.8] -> [345, 396]
    grid_size = [345, 396] 
    
    backbone = PillarRes18BackBone8x2(grid_size=grid_size)
    
    # Mock input
    batch_size = 1
    C = 64 # Input channels (e.g. from VFE)
    # Actually VFE output is [N, C]
    # PillarRes18BackBone8x2 expects pillar_features, pillar_coords, batch_size
    
    # Create dummy pillars
    # Assume 100 pillars
    num_pillars = 100
    # pillar_features: [N, C]
    # In VFE test, output features was [1, 32] (num_filters=[32])
    # Let's assume input C=32
    input_C = 32
    
    # But wait, PillarRes18BackBone8x2 first layer:
    # self.conv1 = nn.Sequential(SparseBasicBlockV(32, 32, ...))
    # It expects 32 channels?
    # SparseBasicBlockV(inplanes, planes, ...)
    # conv0: Conv2d(inplanes, planes)
    # So input features must have 'inplanes' channels.
    # In __init__:
    # SparseBasicBlockV(32, 32, ...)
    # So input channels must be 32.
    
    pillar_features = jt.randn((num_pillars, 32))
    
    # pillar_coords: [N, 4] (batch_idx, z, y, x)
    # Indices must be within grid_size
    # grid_size is [W, H] = [345, 396] ?
    # In __init__: self.sparse_shape = np.array(grid_size)[[1,0]]
    # If grid_size passed is [W, H], then sparse_shape is [H, W].
    # In execute: H, W = self.sparse_shape
    # So H=396, W=345.
    
    # Random coords
    b_idx = jt.zeros((num_pillars, 1)).int()
    z_idx = jt.zeros((num_pillars, 1)).int()
    y_idx = jt.randint(0, 396, (num_pillars, 1)).int()
    x_idx = jt.randint(0, 345, (num_pillars, 1)).int()
    
    pillar_coords = jt.concat([b_idx, z_idx, y_idx, x_idx], dim=1)
    
    print(f"Input features: {pillar_features.shape}")
    print(f"Input coords: {pillar_coords.shape}")
    
    # Run backbone
    outputs = backbone(pillar_features, pillar_coords, batch_size)
    
    # Check outputs
    print("Backbone outputs:")
    for key, value in outputs.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    test_backbone()
