
import numpy as np
import os
from PIL import Image
from projects.mmdet3d_plugin.jittor_adapter import PIPELINES

class RadarPointCloud:
    """Minimal RadarPointCloud implementation to avoid nuscenes-devkit dependency."""
    
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_file(cls, file_name, invalid_states=None, dynprop_states=None, ambig_states=None):
        """Loads RadarPointCloud from a pcd file."""
        meta = {}
        with open(file_name, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip().decode('utf-8')
                if line.startswith('FIELDS'):
                    meta['fields'] = line.split()[1:]
                elif line.startswith('SIZE'):
                    meta['size'] = [int(x) for x in line.split()[1:]]
                elif line.startswith('TYPE'):
                    meta['type'] = line.split()[1:]
                elif line.startswith('COUNT'):
                    meta['count'] = [int(x) for x in line.split()[1:]]
                elif line.startswith('WIDTH'):
                    meta['width'] = int(line.split()[1])
                elif line.startswith('HEIGHT'):
                    meta['height'] = int(line.split()[1])
                elif line.startswith('VIEWPOINT'):
                    meta['viewpoint'] = [float(x) for x in line.split()[1:]]
                elif line.startswith('POINTS'):
                    meta['points'] = int(line.split()[1])
                elif line.startswith('DATA'):
                    meta['data'] = line.split()[1]
                    break
        
        with open(file_name, 'rb') as f:
            data = f.read()
            
        header_end = data.find(b'DATA binary\n')
        if header_end == -1:
             raise ValueError("Not a binary PCD file")
             
        binary_data = data[header_end + len(b'DATA binary\n'):]
        
        num_points = meta['points']
        points = np.frombuffer(binary_data, dtype=np.float32)
        points = points.reshape(18, -1) 
        
        mask = np.ones(points.shape[1], dtype=bool)
        
        if invalid_states is not None:
            invalid = points[14, :]
            for state in invalid_states:
                mask = np.logical_and(mask, invalid != state)
                
        if dynprop_states is not None:
            dynprop = points[3, :]
            sub_mask = np.zeros(points.shape[1], dtype=bool)
            for state in dynprop_states:
                sub_mask = np.logical_or(sub_mask, dynprop == state)
            mask = np.logical_and(mask, sub_mask)
            
        if ambig_states is not None:
             ambig = points[11, :]
             for state in ambig_states:
                 mask = np.logical_and(mask, ambig != state)
                 
        points = points[:, mask]
        return cls(points)


@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File."""
    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=dict(backend='disk')):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename):
        if not os.path.exists(pts_filename):
            return np.zeros((0, self.load_dim), dtype=np.float32)
        try:
            points = np.fromfile(pts_filename, dtype=np.float32)
            points = points.reshape(-1, self.load_dim)
        except Exception:
            points = np.zeros((0, self.load_dim), dtype=np.float32)
        return points

    def __call__(self, results):
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points[:, :self.use_dim]
        results['points'] = points
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps (stub — returns current frame only)."""
    def __init__(self, sweeps_num=10, file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False, remove_close=False, test_mode=False):
        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def __call__(self, results):
        # Stub: just keep current points as-is
        return results


@PIPELINES.register_module(name='LoadRadarPointsFromMultiSweepsV3')
class LoadRadarPointsFromMultiSweeps:
    """Load radar points from multiple sweeps."""
    def __init__(self, sweeps_num=10, file_client_args=dict(backend='disk'),
                 invalid_states=None, dynprop_states=None, ambig_states=None,
                 pad_empty_sweeps=False, remove_close=False, test_mode=False):
        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.invalid_states = invalid_states or [0]
        self.dynprop_states = dynprop_states or list(range(7))
        self.ambig_states = ambig_states or [3]

    def _load_points(self, pts_filename):
        if not os.path.exists(pts_filename):
            return np.zeros((0, 6), dtype=np.float32)
        try:
            raw_points = RadarPointCloud.from_file(
                pts_filename,
                invalid_states=self.invalid_states,
                dynprop_states=self.dynprop_states,
                ambig_states=self.ambig_states).points.T
            xyz = raw_points[:, :3]
            rcs = raw_points[:, 5].reshape(-1, 1)
            vxy_comp = raw_points[:, 8:10]
            points = np.hstack((xyz, rcs, vxy_comp))
        except Exception:
            points = np.zeros((0, 6), dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        if points.shape[0] == 0:
            return points
        x_filt = np.abs(points[:, 0]) < radius
        y_filt = np.abs(points[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        sweep_points_list = []
        if 'radar_sweeps' not in results:
            # No radar sweep info — create empty points
            results['points'] = np.zeros((0, 6), dtype=np.float32)
            return results

        num_sweeps = len(results['radar_sweeps'])
        use_sweeps = min(num_sweeps, self.sweeps_num)
        
        for idx in range(use_sweeps):
            sweep_pts_file_name = results['radar_sweeps'][idx]
            points_sweep = self._load_points(sweep_pts_file_name)
            if points_sweep.shape[0] == 0:
                continue
            if self.remove_close:
                points_sweep = self._remove_close(points_sweep)
            # Time compensation
            if 'radar_sweeps_time_gap' in results:
                points_sweep[:, 0] += points_sweep[:, 4] * results['radar_sweeps_time_gap'][idx]
                points_sweep[:, 1] += points_sweep[:, 5] * results['radar_sweeps_time_gap'][idx]
            if 'radar_sweeps_r2l_rot' in results:
                R = results['radar_sweeps_r2l_rot'][idx]
                T = results['radar_sweeps_r2l_trans'][idx]
                points_sweep[:, :3] = points_sweep[:, :3] @ R
                points_sweep[:, 4:6] = points_sweep[:, 4:6] @ R[:2, :2]
                points_sweep[:, :3] += T
            sweep_points_list.append(points_sweep)
            
        if len(sweep_points_list) > 0:
            points = np.concatenate(sweep_points_list)
        else:
            points = np.zeros((0, 6), dtype=np.float32)
            
        results['points'] = points
        return results

# Also register with original class name
PIPELINES.register_module(name='LoadRadarPointsFromMultiSweeps', module=LoadRadarPointsFromMultiSweeps)


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi-view images from files."""
    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results.get('img_filename', [])
        imgs = []
        for name in filename:
            try:
                # Ensure name is str and use absolute path for safety
                name_str = str(name)
                # On Windows, sometimes relative paths behave weirdly with PIL open directly
                # Using 'with open' ensures file handle is properly managed
                with open(name_str, 'rb') as f:
                    img = Image.open(f)
                    img.load() # Force load image data while file is open
                    img = np.array(img)
            except Exception as e:
                print(f"Error loading image {name}: {e}")
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
        
        results['filename'] = filename
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in imgs] if imgs else []
        results['ori_shape'] = [img.shape for img in imgs] if imgs else []
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadAnnotations3D:
    def __init__(self, with_bbox_3d=True, with_label_3d=True, with_attr_label=False):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def __call__(self, results):
        if 'ann_info' not in results:
            return results
        if self.with_bbox_3d:
            bboxes_3d = results['ann_info']['gt_bboxes_3d']
            if hasattr(bboxes_3d, 'tensor'):
                bboxes_3d = bboxes_3d.tensor.numpy()
            elif hasattr(bboxes_3d, 'numpy'):
                bboxes_3d = bboxes_3d.numpy()
            results['gt_bboxes_3d'] = np.array(bboxes_3d, dtype=np.float32)
            results['bbox3d_fields'] = ['gt_bboxes_3d']
        if self.with_label_3d:
            results['gt_labels_3d'] = np.array(results['ann_info']['gt_labels_3d'], dtype=np.int64)
        return results
