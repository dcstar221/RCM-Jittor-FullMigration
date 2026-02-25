
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
import pickle
import os
import random
import copy
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .jittor_pipelines import Compose

class JittorCustomNuScenesDataset(Dataset):
    def __init__(self, 
                 ann_file,
                 pipeline,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 queue_length=4,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 **kwargs):
        
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality or dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False)
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt
        self.classes = classes
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.queue_length = queue_length
        self.with_velocity = with_velocity
        
        # Load annotations
        self.data_infos = self.load_annotations(self.ann_file)
        
        if self.load_interval > 1:
             self.data_infos = self.data_infos[::self.load_interval]
             
        self.total_len = len(self.data_infos)
        
        # Initialize pipeline
        if isinstance(pipeline, list):
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = pipeline
            
        # Set category map if needed (skipping for now)

    def load_annotations(self, ann_file):
        # Handle relative path for ann_file
        if self.data_root and not os.path.isabs(ann_file):
            ann_file = os.path.join(self.data_root, ann_file)
            
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            data_root=self.data_root
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
                viewpad_[0][2] = 1600 - viewpad_[0][2] # Assuming 1600 width?
                
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
        
        if self.modality['use_radar']:
            if 'radar_sweeps' in info:
                input_dict.update(
                    dict(
                        radar_sweeps = info['radar_sweeps'],
                        radar_sweeps_time_gap=info['radar_sweeps_time_gap'],
                        radar_sweeps_r2l_rot = info['radar_sweeps_r2l_rot'],
                        radar_sweeps_r2l_trans = info['radar_sweeps_r2l_trans']
                    )
                )
            else:
                 # Provide empty defaults if missing but requested
                 input_dict.update(
                    dict(
                        radar_sweeps = [],
                        radar_sweeps_time_gap=[],
                        radar_sweeps_r2l_rot = [],
                        radar_sweeps_r2l_trans = []
                    )
                )

        # Process can_bus
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
        
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        
        gt_names = None
        if 'gt_boxes' in info:
             gt_bboxes_3d = info['gt_boxes']
             gt_names = info['gt_names']
        elif 'gt_bboxes_3d' in info:
             gt_bboxes_3d = info['gt_bboxes_3d']
             gt_labels_3d = info['gt_labels_3d']
        else:
             gt_bboxes_3d = np.zeros((0, 7))
             gt_labels_3d = np.zeros((0,))
             gt_names = []

        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        
        gt_bboxes_3d = gt_bboxes_3d[mask]
        
        if gt_names is not None:
             gt_names = gt_names[mask]
             gt_labels_3d = []
             for cat in gt_names:
                 if cat in self.classes:
                     gt_labels_3d.append(self.classes.index(cat))
                 else:
                     gt_labels_3d.append(-1)
             gt_labels_3d = np.array(gt_labels_3d)
        else:
             gt_labels_3d = gt_labels_3d[mask]
        
        # Filter out -1 labels
        valid_mask = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_mask]
        gt_labels_3d = gt_labels_3d[valid_mask]
        
        ann_info = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d
        )
        return ann_info

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        # img: (N_views, H, W, C) -> stack to (Queue, N_views, H, W, C) ?
        # Actually pipeline 'LoadMultiViewImageFromFiles' puts 'img' as list of arrays
        # 'DefaultFormatBundle3D' might stack them.
        
        # But here 'queue' is list of processed results (dicts).
        # We need to stack 'img' and 'img_metas'.
        
        # Let's look at how nuscenes_dataset_new.py does it.
        # It assumes 'img' is already a DataContainer or tensor?
        # In my pipeline, 'img' is list of numpy arrays (until DefaultFormatBundle3D).
        # If pipeline includes DefaultFormatBundle3D, then 'img' is Jittor array.
        
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

        # Stack images
        # imgs_list elements are Jittor arrays (N, C, H, W)
        # We want (Queue, N, C, H, W) -> (Queue*N, C, H, W) or something?
        # BEVFormer usually expects (B, N, C, H, W) where N is num_cams.
        # With temporal, maybe (B, Queue, N, C, H, W).
        
        # nuscenes_dataset_new.py:
        # queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        
        # So it stacks them.
        # If imgs_list[i] is (N, C, H, W), then stack -> (Queue, N, C, H, W)
        # If we concatenate, we get (Queue*N, C, H, W).
        # BEVFormer usually reshapes later.
        
        # Let's stack on a new dimension 0.
        try:
            queue_img = jt.stack(imgs_list, dim=0)
        except:
             # If input is numpy, stack using numpy then convert
             queue_img = jt.array(np.stack(imgs_list, axis=0))
        
        queue[-1]['img'] = queue_img
        queue[-1]['img_metas'] = metas_map # This is a dict, fine.
        queue = queue[-1]
        return queue

    def prepare_train_data(self, index):
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
        
        # Run pipeline
        example = self.pipeline(input_dict)
        
        if self.filter_empty_gt and \
                (example is None or (example['gt_labels_3d'].numel() == 0)):
            # print(f"DEBUG: Filtering empty GT or None example at index {index}")
            return None
            
        data_queue.insert(0, example)
        
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or (example['gt_labels_3d'].numel() == 0)):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
            
        return self.union2one(data_queue)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)
        while True:
            data = self.prepare_train_data(index)
            if data is None:
                index = random.randint(0, self.total_len - 1)
                continue
            return data

    def __len__(self):
        return self.total_len
    
    def collate_batch(self, batch):
        # Filter None
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return {}
            
        # batch is list of dicts
        # we need to stack tensors and collect metas
        
        # keys: img, points, gt_bboxes_3d, gt_labels_3d, img_metas
        
        # img: (Queue, N, C, H, W) -> stack to (B, Queue, N, C, H, W)
        imgs = []
        points = []
        gt_bboxes = []
        gt_labels = []
        img_metas = []
        
        for item in batch:
            imgs.append(item['img'])
            if 'points' in item:
                points.append(item['points'])
            if 'gt_bboxes_3d' in item:
                gt_bboxes.append(item['gt_bboxes_3d'])
            if 'gt_labels_3d' in item:
                gt_labels.append(item['gt_labels_3d'])
            img_metas.append(item['img_metas'])
            
        res = {}
        try:
            res['img'] = jt.stack(imgs, dim=0)
        except:
             # If imgs contains numpy arrays (e.g. test mode), stack using numpy then convert
             res['img'] = jt.array(np.stack(imgs, axis=0))
             
        res['img_metas'] = img_metas # List of dicts
        
        # Points and GT are usually lists of tensors (because different sizes)
        # Jittor dataset collate might try to stack them if we don't handle it.
        # But we are overriding collate_batch in Jittor?
        # Jittor Dataset has collate_batch argument in set_attrs.
        
        res['points'] = points
        res['gt_bboxes_3d'] = gt_bboxes
        res['gt_labels_3d'] = gt_labels
        
        return res

