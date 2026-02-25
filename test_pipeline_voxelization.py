
import jittor as jt
import numpy as np
import sys
import os
import pickle

# Mock dependencies
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

from projects.mmdet3d_plugin.datasets.jittor_nuscenes import JittorNuScenesDataset, LoadPointsFromFile, DefaultFormatBundle3D, Collect3D
from projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe import DynamicPillarVFESimple2D

# Config
data_root = 'data/nuscenes/'
ann_file = data_root + 'nuscenes_infos_train.pkl'

# Pipeline
pipeline = [
    LoadPointsFromFile(coord_type='LIDAR', load_dim=5, use_dim=5),
    DefaultFormatBundle3D(class_names=['car']),
    Collect3D(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

def test_pipeline_voxelization():
    print("Testing Pipeline + Voxelization...")
    
    if not os.path.exists(ann_file):
        print(f"Annotation file {ann_file} does not exist.")
        return

    dataset = JittorNuScenesDataset(
        ann_file=ann_file,
        data_root=data_root,
        pipeline=pipeline,
        classes=['car'],
        batch_size=1,
        shuffle=False
    )
    
    if len(dataset) > 0:
        item = dataset[0]
        points = item['points'] # [N, 5] jt.Var
        print(f"Loaded points shape: {points.shape}")
        
        # VFE Config
        voxel_size = [0.2, 0.2, 8]
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        grid_size = [345, 396, 1]
        
        vfe = DynamicPillarVFESimple2D(
            num_point_features=5, # 5 dim points
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            num_filters=[32],
            with_cluster_center=True,
            use_absolute_xyz=True
        )
        
        # Prepare input for VFE: list of point tensors
        points_list = [points]
        
        print("Running VFE...")
        features, coors = vfe(points_list)
        
        print(f"VFE Output Features: {features.shape}")
        print(f"VFE Output Coors: {coors.shape}")
        
        if features.shape[0] > 0:
            print("Pipeline + Voxelization Integration Successful!")
        else:
            print("Warning: No features generated (points might be out of range)")
            
            # Check point range
            p_np = points.numpy()
            print("Point Range X:", p_np[:, 0].min(), p_np[:, 0].max())
            print("Point Range Y:", p_np[:, 1].min(), p_np[:, 1].max())
            print("Point Range Z:", p_np[:, 2].min(), p_np[:, 2].max())

if __name__ == "__main__":
    test_pipeline_voxelization()
