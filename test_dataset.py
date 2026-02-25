
import jittor as jt
from projects.mmdet3d_plugin.datasets.jittor_custom_nuscenes_dataset import JittorCustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.jittor_pipelines import LoadRadarPointsFromMultiSweeps, LoadMultiViewImageFromFiles, LoadAnnotations3D, DefaultFormatBundle3D, Collect3D
import os

def test_dataset():
    data_root = '/Volumes/HP P500/data/RCM_Data/v1.0-mini/'
    ann_file = os.path.join(data_root, 'kitti_infos_val_rcmfusion.pkl')
    
    # Check if files exist
    if not os.path.exists(data_root):
        print(f"Data root not found: {data_root}")
        return
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return

    pipeline = [
        LoadRadarPointsFromMultiSweeps(sweeps_num=1),
        LoadMultiViewImageFromFiles(to_float32=True),
        LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True),
        DefaultFormatBundle3D(class_names=['car', 'truck', 'bus']),
        Collect3D(keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

    dataset = JittorCustomNuScenesDataset(
        ann_file=ann_file,
        pipeline=pipeline,
        data_root=data_root,
        load_interval=1,
        classes=['car', 'truck', 'bus'],
        modality=dict(use_lidar=False, use_camera=True, use_radar=True),
        queue_length=2, # Test temporal queue
        batch_size=1,
        shuffle=False,
        num_workers=0, # Important for macOS
        filter_empty_gt=True # Re-enable filtering
    )

    print(f"Dataset length: {len(dataset)}")

    # Try to get one item
    try:
        data = dataset[0]
        print(f"Successfully loaded index 0")
        print(f"Keys: {data.keys()}")
        if 'img' in data:
            print(f"Image shape: {data['img'].shape}")
        if 'points' in data:
            print(f"Points shape: {data['points'].shape}")
            pts_np = data['points'].numpy()
            print(f"Points sample:\n{pts_np}")
        if 'gt_bboxes_3d' in data:
            print(f"GT Bboxes: {data['gt_bboxes_3d'].shape}")
        if 'img_metas' in data:
            print(f"Img Metas keys: {data['img_metas'].keys()}")

        print("\nTesting DataLoader with batch_size=2...")
        dataset.set_attrs(batch_size=2, shuffle=False)
        for batch_idx, batch_data in enumerate(dataset):
            print(f"Batch {batch_idx} loaded")
            print(f"Batch keys: {batch_data.keys()}")
            if 'img' in batch_data:
                # Should be Jittor Var
                print(f"Batch Image shape: {batch_data['img'].shape}")
            if 'points' in batch_data:
                # Should be list
                print(f"Batch Points type: {type(batch_data['points'])}")
                print(f"Batch Points len: {len(batch_data['points'])}")
            if 'gt_bboxes_3d' in batch_data:
                print(f"Batch GT Bboxes type: {type(batch_data['gt_bboxes_3d'])}")
            break
                
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
