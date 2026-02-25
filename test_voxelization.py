
import jittor as jt
import numpy as np
import sys
import os

# Mock dependencies for import
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['mmcv'] = MockModule()
sys.modules['mmdet'] = MockModule()
sys.modules['mmdet.datasets'] = MockModule()
sys.modules['mmdet.datasets.api_wrappers'] = MockModule()
sys.modules['nuscenes'] = MockModule()
sys.modules['nuscenes.eval'] = MockModule()
sys.modules['nuscenes.eval.common'] = MockModule()
sys.modules['nuscenes.eval.common.utils'] = MockModule()
sys.modules['mmdet.datasets'].DATASETS = MockModule()
sys.modules['mmdet.datasets'].CustomDataset = object

from projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe import DynamicPillarVFESimple2D

def test_voxelization():
    print("Testing Voxelization...")
    
    # 1. Create dummy points (N, 5)
    # x, y, z, intensity, ring
    # Make sure some points are within range
    # range: [0, -39.68, -3, 69.12, 39.68, 1]
    points = np.random.rand(100, 5).astype(np.float32)
    points[:, 0] = points[:, 0] * 60 + 5 # x in [5, 65]
    points[:, 1] = points[:, 1] * 70 - 35 # y in [-35, 35]
    points[:, 2] = points[:, 2] * 3 - 2 # z in [-2, 1]
    
    print("Point Range X:", points[:, 0].min(), points[:, 0].max())
    print("Point Range Y:", points[:, 1].min(), points[:, 1].max())
    print("Point Range Z:", points[:, 2].min(), points[:, 2].max())
    
    # Create batch of 2
    points_list = [jt.array(points), jt.array(points)]
    
    # 2. Instantiate VFE
    vfe = DynamicPillarVFESimple2D(
        num_point_features=5, # Match input dim
        voxel_size=[0.2, 0.2, 8],
        grid_size=[512, 512, 1],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        num_filters=[64],
        with_distance=False,
        use_absolute_xyz=True,
        with_cluster_center=True,
        use_norm=True
    )
    
    # 3. Forward
    # Pass list of points
    features, coors = vfe(points_list)
    
    print("Output Features shape:", features.shape)
    print("Output Coors shape:", coors.shape)
    
    # Check coors
    # coors should be (M, 4) -> batch_idx, z, y, x
    coors_np = coors.numpy()
    print("Coors sample:", coors_np[:5])
    
    assert coors.shape[1] == 4
    assert features.shape[0] == coors.shape[0]
    
    # Check if batch indices are correct (0 and 1)
    batch_indices = np.unique(coors_np[:, 0])
    print("Batch indices present:", batch_indices)
    
    print("Voxelization Test Passed!")

if __name__ == "__main__":
    test_voxelization()
