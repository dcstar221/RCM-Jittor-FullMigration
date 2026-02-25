
import jittor as jt
import numpy as np
import sys
import os
import pickle

# Mock mmcv and other dependencies to avoid import errors in __init__.py
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
# Specifically for nuscenes_dataset.py
sys.modules['mmdet.datasets'].DATASETS = MockModule()
sys.modules['mmdet.datasets'].CustomDataset = object

from projects.mmdet3d_plugin.datasets.jittor_nuscenes import JittorNuScenesDataset, LoadPointsFromFile, DefaultFormatBundle3D, Collect3D

# Config
data_root = 'data/nuscenes/'
ann_file = data_root + 'nuscenes_infos_train.pkl'

# Create pipeline
pipeline = [
    LoadPointsFromFile(coord_type='LIDAR', load_dim=5, use_dim=5),
    DefaultFormatBundle3D(class_names=['car']),
    Collect3D(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

def test_dataset():
    print("Testing Jittor Dataset Loading...")
    
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
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Fetch item using __getitem__ directly (or iterator)
        # Jittor dataset supports indexing? Yes usually.
        # But Jittor dataset is designed for DataLoader iteration.
        # Let's try direct access first
        try:
            item = dataset[0]
            if item is None:
                print("Item is None")
            else:
                print("Item loaded keys:", item.keys())
                if 'points' in item:
                    print("Points shape:", item['points'].shape)
                    print("Points type:", type(item['points']))
                if 'gt_bboxes_3d' in item:
                    print("GT Bboxes:", item['gt_bboxes_3d'])
        except Exception as e:
            print(f"Error accessing item 0: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
