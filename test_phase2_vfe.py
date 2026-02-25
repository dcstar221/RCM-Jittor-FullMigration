
import jittor as jt
import numpy as np
from projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe import DynamicPillarVFESimple2D

def test_vfe():
    print("Testing DynamicPillarVFESimple2D...")
    
    # Initialize VFE
    vfe = DynamicPillarVFESimple2D(
        num_point_features=4,
        voxel_size=[0.2, 0.2, 8],
        grid_size=[512, 512, 1],
        point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
        num_filters=[64],
        with_distance=False,
        use_absolute_xyz=True,
        with_cluster_center=True,
        use_norm=True
    )
    
    # Create dummy input
    # 100 points with 4 features (x, y, z, intensity)
    num_points = 100
    features = jt.rand(num_points, 4)
    # 100 points with 4 coordinates (batch_idx, z, y, x) - although VFE usually takes (batch_idx, z, y, x)
    # The VFE expects features and coors. 
    # In mmdet3d, coors are usually (batch_idx, z, y, x) for voxels, or just (batch_idx, z, y, x) for points?
    # Wait, DynamicPillarVFE usually takes (features, coors).
    # Let's check the implementation of execute in dynamic_pillar_vfe.py
    
    # Based on my previous read of dynamic_pillar_vfe.py (implied):
    # It likely uses scatter_mean/max on features based on coors.
    
    # Let's assume coors are (batch_idx, z_idx, y_idx, x_idx)
    coors = jt.zeros((num_points, 4), dtype=jt.int32)
    # Randomly assign to some pillars
    coors[:, 3] = jt.randint(0, 512, (num_points,)) # x
    coors[:, 2] = jt.randint(0, 512, (num_points,)) # y
    coors[:, 1] = 0 # z (usually 0 for pillars)
    coors[:, 0] = 0 # batch 0
    
    print("Input features shape:", features.shape)
    print("Input coors shape:", coors.shape)
    
    # Run VFE
    output_features, output_coors = vfe(features, coors)
    
    print("Output features shape:", output_features.shape)
    print("Output coors shape:", output_coors.shape)
    
    # Basic checks
    assert output_features.shape[1] == 64, "Output features should have 64 channels"
    assert output_features.shape[0] == output_coors.shape[0], "Output features and coors should have same length"
    
    print("DynamicPillarVFESimple2D test passed!")

if __name__ == "__main__":
    jt.flags.use_cuda = 0 # Force CPU for testing if needed, or let it decide
    test_vfe()
