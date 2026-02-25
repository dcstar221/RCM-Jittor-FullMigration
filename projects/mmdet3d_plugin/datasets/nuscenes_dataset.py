import copy
import numpy as np
import random
import mmcv
from os import path as osp
import jittor as jt
from jittor.dataset import Dataset

from projects.mmdet3d_plugin.jittor_adapter import DATASETS, PIPELINES, build_from_cfg
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.datasets.api_wrappers import COCO
from .nuscnes_eval import NuScenesEval_custom

# Simple Compose implementation
class Compose:
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

@DATASETS.register_module()
class CustomNuScenesDataset(Dataset):
    r"""NuScenes Dataset.
    
    Jittor implementation of CustomNuScenesDataset.
    """
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(self, 
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 eval_version='detection_cvpr_2019',
                 queue_length=4, 
                 bev_size=(200, 200), 
                 overlap_test=False, 
                 with_info=False,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 **kwargs):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d = box_type_3d
        self.classes = classes or self.CLASSES
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.eval_version = eval_version
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.overlap_test = overlap_test
        self.with_info = with_info
        
        # Load annotations
        self.data_infos = self.load_annotations(self.ann_file)
        self.total_len = len(self.data_infos)
        
        # Pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
            
        # Init evaluation configs
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        
    def load_annotations(self, ann_file):
        """Load annotations from ann_file."""
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_ann_info(self, index):
        """Get annotation info according to the given index."""
        info = self.data_infos[index]
        if 'gt_bboxes_3d' in info:
             return dict(
                 gt_bboxes_3d=info['gt_bboxes_3d'],
                 gt_labels_3d=info['gt_labels_3d'],
                 gt_names=info['gt_names']
             )
        return None

    def pre_pipeline(self, results):
        """Initialization before data preparation."""
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = 0 # LiDAR

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]
        
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            radar_timestamp=info.get('radar_timestamp', 0),
        )
           
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_intrinsics_flips = []
            cam_ts = []
            l2c_rs_=[]
            
            for cam_type, cam_info in info['cams'].items():
                cam_trans = cam_info['sensor2lidar_translation']
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_trans @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                viewpad_ = viewpad.copy()
                viewpad_[0][2] = 1600 - viewpad_[0][2]
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                cam_intrinsics_flips.append(viewpad_)
                cam_ts.append(cam_trans)
                l2c_rs_.append(lidar2cam_r)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsics_flip=cam_intrinsics_flips,
                    cam_t = np.array(cam_ts),
                    l2c_r_= np.array(l2c_rs_)
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        if self.modality.get('use_radar', False):
            input_dict.update(
                dict(
                    radar_pts_filename=info.get('radar_path', ''),
                    radar_sweeps = info.get('RADAR_sweeps', [])
                )
            )
            
        # Handle can_bus
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        
        if info.get('gt_bboxes_2d') is not None:
            gt_bboxes = info['gt_bboxes_2d']
            gt_labels = info['gt_labels_2d']
            input_dict.update(
                dict(
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels
                )
            )
        
        return input_dict

    def union2one(self, queue):
        """convert sample queue into one single sample."""
        # Replace DataContainer with direct assignment
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        # In Jittor pipeline, imgs are already numpy or jt.array, so we can stack them
        # If they are numpy, we stack to numpy. If jt, stack to jt.
        # Assuming they are numpy arrays from pipeline
        if isinstance(imgs_list[0], np.ndarray):
             queue[-1]['img'] = np.stack(imgs_list)
        else:
             queue[-1]['img'] = jt.stack(imgs_list)
             
        queue[-1]['img_metas'] = metas_map
        queue = queue[-1]
        return queue

    def prepare_train_data(self, index):
        """Training data preparation."""
        data_queue = []
        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d'] != -1).any()):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d'] != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def prepare_test_data(self, index):
        """Prepare data for testing."""
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def _rand_another(self, idx):
        """Randomly get another item."""
        return np.random.randint(0, len(self.data_infos))

    def __getitem__(self, idx):
        """Get item from infos according to the given index."""
        # macOS index protection
        if self.total_len > 0:
            idx = idx % self.total_len
            
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def collate_batch(self, batch):
        """Custom collate function for Jittor."""
        collated = {}
        
        if 'img' in batch[0]:
            # batch[i]['img'] is (N_views, C, H, W) or (Queue, N_views, C, H, W)
            # We want to stack them
            imgs = [b['img'] for b in batch]
            if isinstance(imgs[0], np.ndarray):
                collated['img'] = jt.array(np.stack(imgs))
            else:
                collated['img'] = jt.stack(imgs)
            
        if 'points' in batch[0]:
            collated['points'] = [jt.array(b['points']) for b in batch]
            
        if 'gt_bboxes_3d' in batch[0]:
            collated['gt_bboxes_3d'] = [b['gt_bboxes_3d'].tensor if hasattr(b['gt_bboxes_3d'], 'tensor') else jt.array(b['gt_bboxes_3d']) for b in batch]
            
        if 'gt_labels_3d' in batch[0]:
            collated['gt_labels_3d'] = [jt.array(b['gt_labels_3d']) for b in batch]
            
        metas = []
        for b in batch:
            meta = {}
            for k in b:
                if k not in ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']:
                    meta[k] = b[k]
            metas.append(meta)
        collated['img_metas'] = metas
        
        return collated
