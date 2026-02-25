
import numpy as np
import os
import jittor as jt
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for i, t in enumerate(self.transforms):
            print(f"Running transform {i}: {type(t).__name__}")
            data = t(data)
            if data is None:
                return None
        return data

class LoadMultiViewImageFromFiles:
    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views) -> (num_views, h, w, c)
        imgs = []
        for name in filename:
            # Handle path joining if needed
            if 'data_root' in results and not os.path.isabs(name):
                name = os.path.join(results['data_root'], name)
            
            # Use PIL or cv2 (if available). standard PIL is safer
            img = Image.open(name)
            img = np.array(img)
            
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
            
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in imgs]
        results['ori_shape'] = [img.shape for img in imgs]
        # Set initial values for default meta_keys
        results['pad_shape'] = [img.shape for img in imgs]
        results['scale_factor'] = 1.0
        return results

class RadarPointCloud:
    """Minimal RadarPointCloud implementation to avoid nuscenes-devkit dependency."""
    
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_file(cls, file_name, invalid_states=None, dynprop_states=None, ambig_states=None):
        """Loads RadarPointCloud from a pcd file."""
        meta = {}
        # Read header
        if not os.path.exists(file_name):
             print(f"DEBUG: PC File not found: {file_name}")
             return cls(np.zeros((18, 0), dtype=np.float32))

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
        
        # Read binary data
        with open(file_name, 'rb') as f:
            data = f.read()
            
        header_end = data.find(b'DATA binary\n')
        if header_end == -1:
             print(f"DEBUG: No DATA binary found in {file_name}")
             # Fallback for some files? Or raise error.
             # Assuming standard pcd format
             raise ValueError(f"Not a binary PCD file: {file_name}")
             
        binary_data = data[header_end + len(b'DATA binary\n'):]
        
        num_points = meta['points']
        
        # Construct structured dtype from header
        dtype_list = []
        for i, field_name in enumerate(meta['fields']):
            type_char = meta['type'][i]
            size_bytes = meta['size'][i]
            
            # Map PCD type/size to numpy dtype
            if type_char == 'F':
                if size_bytes == 4:
                    dt = 'f4'
                elif size_bytes == 8:
                    dt = 'f8'
                else:
                    raise ValueError(f"Unknown float size: {size_bytes}")
            elif type_char == 'I':
                if size_bytes == 1:
                    dt = 'i1'
                elif size_bytes == 2:
                    dt = 'i2'
                elif size_bytes == 4:
                    dt = 'i4'
                elif size_bytes == 8:
                    dt = 'i8'
                else:
                    raise ValueError(f"Unknown int size: {size_bytes}")
            elif type_char == 'U':
                if size_bytes == 1:
                    dt = 'u1'
                elif size_bytes == 2:
                    dt = 'u2'
                elif size_bytes == 4:
                    dt = 'u4'
                elif size_bytes == 8:
                    dt = 'u8'
                else:
                    raise ValueError(f"Unknown uint size: {size_bytes}")
            else:
                 # Fallback to float if unknown
                 dt = 'f4'
            
            dtype_list.append((field_name, dt))
            
        try:
            # Read structured array
            pc_data = np.frombuffer(binary_data, dtype=dtype_list, count=num_points)
            
            # Convert to standard (18, N) float32 array
            # We iterate fields in order to match expected 18 channels
            points_out = np.zeros((len(meta['fields']), num_points), dtype=np.float32)
            
            for i, field_name in enumerate(meta['fields']):
                points_out[i, :] = pc_data[field_name].astype(np.float32)
            
            points = points_out # (18, N)
            
            mask = np.ones(points.shape[1], dtype=bool)
            
            if invalid_states is not None:
                # invalid_state is index 14
                invalid = points[14, :]
                # No cast needed, already float, but comparison works
                for state in invalid_states:
                    mask = np.logical_and(mask, invalid != state)
                    
            if dynprop_states is not None:
                # dyn_prop is index 3
                dynprop = points[3, :]
                sub_mask = np.zeros(points.shape[1], dtype=bool)
                for state in dynprop_states:
                    sub_mask = np.logical_or(sub_mask, dynprop == state)
                mask = np.logical_and(mask, sub_mask)
                
            if ambig_states is not None:
                # ambig_state is index 11
                ambig = points[11, :]
                for state in ambig_states:
                    mask = np.logical_and(mask, ambig != state)
                
            points_filtered = points[:, mask]
            return cls(points_filtered)
            
        except Exception as e:
            print(f"DEBUG: Failed to read as int32/float32: {e}")
            return cls(np.zeros((18, 0), dtype=np.float32))

class LoadRadarPointsFromMultiSweeps:
    """Load radar points from multiple sweeps."""
    def __init__(self, sweeps_num=10, 
                 invalid_states=None, dynprop_states=None, ambig_states=None,
                 remove_close=False, test_mode=False, file_client_args=None):
        self.sweeps_num = sweeps_num
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.invalid_states = invalid_states or [0]
        self.dynprop_states = dynprop_states or list(range(7))
        self.ambig_states = ambig_states or [3]
        self.file_client_args = file_client_args

    def _load_points(self, pts_filename, data_root=None):
        if data_root and not os.path.isabs(pts_filename):
            pts_filename = os.path.join(data_root, pts_filename)
            
        if not os.path.exists(pts_filename):
            print(f"DEBUG: File not found: {pts_filename}")
            return np.zeros((0, 6), dtype=np.float32)
        try:
            # print(f"DEBUG: Loading {pts_filename}")
            pc = RadarPointCloud.from_file(
                pts_filename,
                invalid_states=self.invalid_states,
                dynprop_states=self.dynprop_states,
                ambig_states=self.ambig_states)
            
            # print(f"DEBUG: Loaded PC shape: {pc.points.shape}")
            raw_points = pc.points.T
            # T makes it (N, 18)
            
            xyz = raw_points[:, :3]
            rcs = raw_points[:, 5].reshape(-1, 1)
            vxy_comp = raw_points[:, 8:10]
            points = np.hstack((xyz, rcs, vxy_comp))
        except Exception as e:
            print(f"Error loading {pts_filename}: {e}")
            import traceback
            traceback.print_exc()
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
        data_root = results.get('data_root', None)
        
        if 'radar_sweeps' not in results:
            results['points'] = np.zeros((0, 6), dtype=np.float32)
            return results

        num_sweeps = len(results['radar_sweeps'])
        use_sweeps = min(num_sweeps, self.sweeps_num)
        
        for idx in range(use_sweeps):
            sweep_pts_file_name = results['radar_sweeps'][idx]
            points_sweep = self._load_points(sweep_pts_file_name, data_root)
            if points_sweep.shape[0] == 0:
                continue
            if self.remove_close:
                points_sweep = self._remove_close(points_sweep)
            # Time compensation
            if 'radar_sweeps_time_gap' in results:
                gap = results['radar_sweeps_time_gap'][idx]
                points_sweep[:, 0] += points_sweep[:, 4] * gap
                points_sweep[:, 1] += points_sweep[:, 5] * gap
            if 'radar_sweeps_r2l_rot' in results:
                R = results['radar_sweeps_r2l_rot'][idx]
                T = results['radar_sweeps_r2l_trans'][idx]
                # Rotate xyz
                points_sweep[:, :3] = points_sweep[:, :3] @ R
                # Rotate velocity (v_x, v_y)
                points_sweep[:, 4:6] = points_sweep[:, 4:6] @ R[:2, :2]
                # Translate
                points_sweep[:, :3] += T
            sweep_points_list.append(points_sweep)
            
        if len(sweep_points_list) > 0:
            points = np.concatenate(sweep_points_list)
        else:
            points = np.zeros((0, 6), dtype=np.float32)
            
        results['points'] = points
        return results

class LoadAnnotations3D:
    def __init__(self, with_bbox_3d=True, with_label_3d=True, with_attr_label=False):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def __call__(self, results):
        if self.with_bbox_3d:
            results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
            # Note: We assume gt_bboxes_3d is already a numpy array or tensor from get_ann_info
            
        if self.with_label_3d:
            results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
            
        return results

class DefaultFormatBundle3D:
    def __init__(self, class_names, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        if 'points' in results:
            results['points'] = jt.array(results['points'])
        if 'gt_bboxes_3d' in results:
            # Check if it's already a tensor or numpy
            if isinstance(results['gt_bboxes_3d'], np.ndarray):
                results['gt_bboxes_3d'] = jt.array(results['gt_bboxes_3d'])
            # If it's a LiDARRotatedBox or similar, might need special handling
            # For now assume numpy array
        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = jt.array(results['gt_labels_3d'])
            
        if 'img' in results:
            # img is list of numpy arrays (H, W, C)
            # Stack them: (N, H, W, C)
            imgs = np.stack(results['img'], axis=0)
            # Convert to (N, C, H, W) for PyTorch/Jittor convention? 
            # Usually pipelines do Normalize -> Pad -> Transpose
            # We missed Normalize/Pad in this minimal pipeline
            # Let's assume we want (N, C, H, W)
            imgs = imgs.transpose(0, 3, 1, 2)
            results['img'] = jt.array(imgs)
            
        return results

class Collect3D:
    def __init__(self, keys, meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'can_bus')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        
        # Collect meta info
        if 'img_metas' not in data:
            data['img_metas'] = {}
            
        for key in self.meta_keys:
            if key in results:
                data['img_metas'][key] = results[key]
                
        return data

# Aliases for compatibility with config
LoadRadarPointsFromMultiSweepsV3 = LoadRadarPointsFromMultiSweeps
CustomCollect3D = Collect3D
