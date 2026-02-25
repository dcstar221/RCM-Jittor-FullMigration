import jittor as jt
import jittor.nn as nn
import numpy as np
from projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe import DynamicPillarVFESimple2D

jt.flags.use_cuda = 0 # Use CPU

def test_vfe():
    print("Testing DynamicPillarVFESimple2D with num_point_features=6...")
    
    voxel_size = [0.075, 0.075, 0.2]
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    grid_size = [128, 128, 40] # Same as verify_model_forward.py now
    
    vfe = DynamicPillarVFESimple2D(
        num_point_features=6, # x, y, z, intensity, elongation, timestamp
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range,
        num_filters=[32],
        with_distance=False,
        use_absolute_xyz=True,
        with_cluster_center=True,
        legacy=False
    )
    
    # Create dummy points
    # 500 points, 6 features
    points = jt.randn(500, 6) 
    # Adjust range to be within point_cloud_range
    points[:, 0] = points[:, 0] * 50 # x
    points[:, 1] = points[:, 1] * 50 # y
    points[:, 2] = points[:, 2] * 2 # z
    
    print("Input points shape:", points.shape)
    
    # Pass as list of points (batch size 1)
    features, coors = vfe([points])
    
    print("VFE output features shape:", features.shape)
    print("VFE output coors shape:", coors.shape)
    
    # Trigger execution
    features.sync()
    coors.sync()
    print("Execution complete.")

if __name__ == "__main__":
    test_vfe()
