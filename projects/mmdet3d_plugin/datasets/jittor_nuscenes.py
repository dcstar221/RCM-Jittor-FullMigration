
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
import pickle
import os
from PIL import Image

class JittorNuScenesDataset(Dataset):
    def __init__(self, 
                 ann_file,
                 data_root,
                 pipeline=None,
                 classes=None,
                 load_interval=1,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 **kwargs):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.ann_file = ann_file
        self.classes = classes
        self.load_interval = load_interval
        self.pipeline = pipeline
        
        # Load annotations
        self.data_infos = self.load_annotations(self.ann_file)
        # Apply load interval
        self.data_infos = self.data_infos[::self.load_interval]
        self.total_len = len(self.data_infos)
        
    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        return data_infos

    def __getitem__(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        
        # Add GT info if available
        if 'gt_bboxes_3d' in info:
            input_dict['gt_bboxes_3d'] = info['gt_bboxes_3d']
            input_dict['gt_labels_3d'] = info['gt_labels_3d']

        # Process pipeline
        if self.pipeline:
            for transform in self.pipeline:
                input_dict = transform(input_dict)
                if input_dict is None:
                    return None
                    
        return input_dict

    def __len__(self):
        return self.total_len

# Define basic transforms needed for the pipeline
class LoadPointsFromFile:
    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=None):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        pts_filename = results['pts_filename']
        # Handle relative paths if needed
        # Assuming pts_filename is relative to project root or absolute
        if not os.path.exists(pts_filename):
            # Try prepending data_root if available in results? No, usually passed in filename
            # For now, just try to load
            print(f"Warning: File not found {pts_filename}")
            points = np.zeros((0, self.load_dim), dtype=np.float32)
        else:
            try:
                points = np.fromfile(pts_filename, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
            except Exception as e:
                print(f"Error loading {pts_filename}: {e}")
                points = np.zeros((0, self.load_dim), dtype=np.float32)
        
        points = points[:, :self.use_dim]
        results['points'] = points
        return results

class DefaultFormatBundle3D:
    def __init__(self, class_names=None):
        self.class_names = class_names

    def __call__(self, results):
        if 'points' in results:
            results['points'] = jt.array(results['points'])
        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'] = jt.array(results['gt_bboxes_3d'])
        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = jt.array(results['gt_labels_3d'])
        return results

class Collect3D:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        return data

