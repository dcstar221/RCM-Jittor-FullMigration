
import jittor as jt
import sys
import os
import numpy as np

# Add current directory to path to ensure projects module is found
sys.path.append(os.getcwd())

from projects.mmdet3d_plugin.datasets.jittor_custom_nuscenes_dataset import JittorCustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.jittor_pipelines import LoadRadarPointsFromMultiSweeps, LoadMultiViewImageFromFiles, LoadAnnotations3D, DefaultFormatBundle3D, Collect3D

def verify_dataset():
    print("Starting Dataset Verification...")
    
    # Path configuration
    data_root = 'data/nuscenes'
    # ann_file_path is the actual path to check existence
    ann_file_path = os.path.join(data_root, 'kitti_infos_val_rcmfusion.pkl')
    # ann_file_arg is the argument passed to the dataset (relative to data_root)
    ann_file_arg = 'kitti_infos_val_rcmfusion.pkl'
    
    # Check if files exist
    if not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        return
    if not os.path.exists(ann_file_path):
        print(f"❌ Annotation file not found: {ann_file_path}")
        return

    print(f"✅ Data root: {data_root}")
    print(f"✅ Annotation file: {ann_file_path}")

    # Define a simple pipeline for verification
    pipeline = [
        LoadRadarPointsFromMultiSweeps(sweeps_num=1, test_mode=True),
        LoadMultiViewImageFromFiles(to_float32=True),
        # LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True), # Skip for now to focus on data loading
        # DefaultFormatBundle3D(class_names=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']),
        # Collect3D(keys=['img', 'points'])
    ]

    print("Initializing JittorCustomNuScenesDataset...")
    try:
        dataset = JittorCustomNuScenesDataset(
            ann_file=ann_file_arg,
            pipeline=pipeline,
            data_root=data_root,
            load_interval=1,
            classes=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
            modality=dict(use_lidar=False, use_camera=True, use_radar=True, use_map=False, use_external=False),
            queue_length=1, 
            batch_size=1,
            shuffle=False,
            num_workers=0,
            test_mode=True # Use test mode to avoid complex training pipeline logic for now
        )
        print(f"✅ Dataset initialized. Length: {len(dataset)}")
    except Exception as e:
        print(f"❌ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try to get one item
    print("\nAttempting to load the first sample...")
    try:
        data = dataset[0]
        print(f"✅ Successfully loaded index 0")
        print(f"Keys available: {data.keys()}")
        
        # Check Image
        if 'img' in data:
            # LoadMultiViewImageFromFiles returns list of numpy arrays if DefaultFormatBundle3D is not used
            imgs = data['img']
            if isinstance(imgs, list):
                print(f"✅ Image loaded. Type: List of {len(imgs)} arrays.")
                print(f"   Shape of first view: {imgs[0].shape}")
            else:
                print(f"✅ Image loaded. Shape: {imgs.shape}")
        else:
            print("❌ 'img' key missing in data")

        # Check Points
        if 'points' in data:
            points = data['points']
            print(f"✅ Radar Points loaded. Shape: {points.shape}")
            if points.shape[0] > 0:
                print(f"   Sample point (first 5 features): {points[0, :5]}")
            else:
                print("   ⚠️ Points array is empty (might be expected for some frames)")
        else:
            print("❌ 'points' key missing in data")
            
        # Check Meta
        if 'img_metas' in data:
            print(f"✅ Image Metas present.")
            # print(f"   Metas keys: {data['img_metas'].keys()}")

    except Exception as e:
        print(f"❌ Error loading data item: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
