
import os
import sys

# Set env vars to prevent OpenMP deadlocks BEFORE importing anything
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Step 1: Importing jittor (first to prevent OpenMP deadlock)...", flush=True)
import jittor as jt
# Force Jittor flags
jt.flags.use_cuda = 0
jt.flags.use_mkl = 0

print("Step 2: Importing numpy...", flush=True)
import numpy as np

print("Step 3: Importing CustomNuScenesDataset...", flush=True)
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
print("Step 4: Importing PIPELINES...", flush=True)
from projects.mmdet3d_plugin.jittor_adapter import PIPELINES, build_from_cfg
print("Step 5: Importing datasets.pipelines...", flush=True)
import projects.mmdet3d_plugin.datasets.pipelines  # Ensure pipelines are registered
print("Imports done.", flush=True)

# Mock the dataset to avoid needing real files
class MockNuScenesDataset(CustomNuScenesDataset):
    def __init__(self, **kwargs):
        # Skip super init that loads files
        self.data_root = "tmp_data"
        self.ann_file = "tmp.pkl"
        self.test_mode = False
        self.modality = dict(use_lidar=True, use_camera=True)
        self.box_type_3d = 'LiDAR'
        self.filter_empty_gt = False
        self.classes = ['car', 'truck']
        self.load_interval = 1
        self.with_velocity = True
        self.use_valid_flag = False
        self.eval_version = 'detection_cvpr_2019'
        self.queue_length = 3
        self.bev_size = (200, 200)
        self.overlap_test = False
        self.with_info = False
        
        # Mock data infos
        self.data_infos = []
        for i in range(10):
            self.data_infos.append({
                'token': f'token_{i}',
                'lidar_path': f'lidar_{i}.bin',
                'sweeps': [],
                'lidar2ego_translation': [0, 0, 0],
                'lidar2ego_rotation': [1, 0, 0, 0],
                'ego2global_translation': [i, 0, 0],
                'ego2global_rotation': [1, 0, 0, 0],
                'prev': f'token_{i-1}' if i > 0 else '',
                'next': f'token_{i+1}' if i < 9 else '',
                'scene_token': 'scene_0',
                'can_bus': np.zeros(18),
                'frame_idx': i,
                'timestamp': i * 0.5 * 1e6,
                'cams': {
                    'CAM_FRONT': {
                        'data_path': f'cam_front_{i}.jpg',
                        'sensor2lidar_translation': [0, 0, 0],
                        'sensor2lidar_rotation': np.eye(3),
                        'cam_intrinsic': np.eye(3)
                    }
                },
                'gt_bboxes_3d': np.array([[10, 0, 0, 2, 2, 2, 0, 0, 0]], dtype=np.float32),
                'gt_labels_3d': np.array([0], dtype=np.int64),
                'gt_names': ['car']
            })
        self.total_len = len(self.data_infos)
        
        # Build pipeline
        pipeline_cfg = kwargs.get('pipeline', [])
        if pipeline_cfg:
            self.pipeline = build_from_cfg(dict(type='Compose', transforms=pipeline_cfg), PIPELINES)

    def load_annotations(self, ann_file):
        return self.data_infos

# Mock file loading to return random data
class MockLoadPointsFromFile:
    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=dict(backend='disk')):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        results['points'] = np.random.rand(100, self.use_dim).astype(np.float32)
        return results

class MockLoadMultiViewImageFromFiles:
    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filenames = results['img_filename']
        imgs = []
        for _ in filenames:
            imgs.append(np.random.rand(256, 512, 3).astype(np.float32))
        results['img'] = imgs
        results['img_shape'] = [(256, 512, 3)] * len(filenames)
        results['ori_shape'] = [(256, 512, 3)] * len(filenames)
        return results

# Register mocks
PIPELINES.register_module(module=MockLoadPointsFromFile, name='LoadPointsFromFile', force=True)
PIPELINES.register_module(module=MockLoadMultiViewImageFromFiles, name='LoadMultiViewImageFromFiles', force=True)

def test_pipeline():
    print("Testing Pipeline...")
    
    pipeline_config = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
        dict(type='LoadMultiViewImageFromFiles'),
        dict(type='PointsRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        dict(type='ObjectNameFilter', classes=['car', 'truck']),
        dict(type='DefaultFormatBundle3D', class_names=['car', 'truck']),
        dict(type='CustomCollect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    
    dataset = MockNuScenesDataset(pipeline=pipeline_config)
    
    print("Fetching item 5...")
    data = dataset[5]
    
    print("Data keys:", data.keys())
    if 'img' in data:
        print("Image shape:", data['img'].shape) # Expected: (Queue, Views, 3, H, W)
    if 'points' in data:
        print("Points shape:", data['points'][0].shape) # List of points
    
    # Test collate
    print("Testing collate_batch...")
    batch = [dataset[5], dataset[6]]
    collated = dataset.collate_batch(batch)
    
    if 'img' in collated:
        print("Collated Image shape:", collated['img'].shape) # Expected: (Batch, Queue, Views, 3, H, W)
        print("Collated Image type:", type(collated['img']))
        
    if 'gt_bboxes_3d' in collated:
        print("Collated Boxes type:", type(collated['gt_bboxes_3d']))
        print("Collated Boxes length:", len(collated['gt_bboxes_3d']))
        
    print("Pipeline test passed!")

if __name__ == "__main__":
    test_pipeline()
