
import sys
import os
import jittor as jt
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

# Import necessary modules to trigger registration
# Ensure all Jittor modules are imported
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.jittor_adapter import DETECTORS, build_detector

# Explicitly import modules to ensure registration
import projects.mmdet3d_plugin.rcm_fusion.detectors.rcm_fusion_jittor
import projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe
import projects.mmdet3d_plugin.models.backbones.resnet_jittor
import projects.mmdet3d_plugin.models.backbones.spconv_backbone_2d
import projects.mmdet3d_plugin.models.necks.fpn_jittor
import projects.mmdet3d_plugin.models.necks.second_fpn
import projects.mmdet3d_plugin.rcm_fusion.dense_heads.feature_level_fusion
import projects.mmdet3d_plugin.models.fusion_layers.instance_level_fusion_jittor
import projects.mmdet3d_plugin.rcm_fusion.modules.radar_guided_bev_encoder
import projects.mmdet3d_plugin.rcm_fusion.modules.transformer_radar
import projects.mmdet3d_plugin.rcm_fusion.modules.radar_guided_bev_attention
import projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder

def main():
    print("Starting verification...")
    jt.flags.use_cuda = 0 # Use CPU for basic check if possible, or 1 if GPU needed
    # jt.flags.use_cuda = 1 

    # Mock Config
    # Based on rcm-fusion_r50.py
    voxel_size = [0.8, 0.8, 0.2] # Adjusted for 128x128 grid (102.4m / 128 = 0.8m)
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    grid_size = [128, 128, 40] # Smaller grid for faster verification
    
    model_cfg = dict(
        type='RCMFusionJittor', # Use Jittor class
        freeze_img=False,
        freeze_pts=False,
        use_grid_mask=True,
        video_test_mode=False,
        video_train_mode=False,
        pts_voxel_encoder=dict(
            type='DynamicPillarVFESimple2D',
            num_point_features=6, # x, y, z, intensity, elongation, timestamp
            voxel_size=voxel_size,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            num_filters=[32],
            with_distance=False,
            use_absolute_xyz=True,
            with_cluster_center=True,
            legacy=False),
        pts_backbone=dict(
            type='PillarRes18BackBone8x2',
            grid_size=grid_size),
        pts_neck=dict(
            type='BaseBEVBackboneV1',
            layer_nums=[3, 3],
            num_filters=[256, 256],
            upsample_strides=[1, 2],
            num_upsample_filters=[128, 128]),
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=0,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            with_cp=False), # pretrained removed
        img_neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=4,
            relu_before_extra_convs=True),
        pts_bbox_head=dict(
            type='FeatureLevelFusion',
            bev_h=16, # Reduced for test
            bev_w=16, # Reduced for test
            num_query=900,
            num_classes=10,
            embed_dims=256, # Explicitly added
            # in_channels=256,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type='PerceptionTransformerRadar',
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=256,
                encoder=dict(
                    type='RadarGuidedBEVEncoder',
                    num_layers=1, # Reduced for test
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='RadarGuidedBEVEncoderLayer',
                        attn_cfgs=[
                            dict(
                                type='RadarGuidedBEVAttention',
                                embed_dims=256,
                                num_levels=1),
                            dict(
                                type='RadarGuidedBEVAttention',
                                embed_dims=256,
                                num_levels=4)
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm'))),
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=1, # Reduced for test
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                    type='CustomMSDeformableAttention',
                                    embed_dims=256,
                                    num_levels=1,
                                    batch_first=True)
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')))),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
            # loss_iou=dict(type='GIoULoss', loss_weight=0.0)
            ),
        pts_fusion_layer=dict( 
            type='InstanceLevelFusion',
            radii=(0.8, 1.6, 3.2),
            num_samples=(16, 16, 16),
            sa_mlps=((32, 32, 64), (32, 32, 64), (32, 32, 64)),
            dilated_group=(False, False, False),
            norm_cfg=dict(type='BN2d'),
            sa_cfg=dict(
                type='Detr3DCrossAtten',
                pc_range=point_cloud_range,
                num_points=1,
                embed_dims=256,
                num_levels=1),
            grid_size=[16, 16], # bev_h, bev_w, reduced for test
            grid_fps=1024,
            code_size=10,
            num_classes=10,
            input_channels=256,
        )
    )

    # Build model
    print("Building model...")
    try:
        model = build_detector(model_cfg)
        print("Model built successfully!")
    except Exception as e:
        print(f"Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dummy input
    B = 1
    N_cam = 6
    C_img = 3
    H_img = 256
    W_img = 704
    
    img = jt.randn((B, N_cam, C_img, H_img, W_img))
    points = [jt.randn((500, 6))] # 500 points, 6 features
    
    img_metas = [{
        'filename': 'dummy_file.jpg',
        'ori_shape': (900, 1600, 3),
        'img_shape': [(H_img, W_img, 3)] * N_cam,
        'lidar2img': [np.eye(4).tolist()] * N_cam,
        'pcd_horizontal_flip': False,
        'pcd_vertical_flip': False,
        'box_type_3d': 'LiDAR',
        'box_mode_3d': 0,
        'pcd_rotation': np.eye(3).tolist(),
        'pcd_scale_factor': 1.0,
        'pcd_trans': np.zeros(3).tolist(),
        'sample_idx': 0,
        'pts_filename': 'dummy_pts.bin',
        'scene_token': 'dummy_scene_token',
        'can_bus': np.zeros(18),
    }] * B
    
    print("Running forward_test...")
    
    # Debugging steps
    print("Debug: Extracting image features...")
    try:
        img_feats = model.extract_img_feat(img, img_metas)
        print("Image features extracted. Shape:", [f.shape for f in img_feats] if isinstance(img_feats, (list, tuple)) else img_feats.shape)
    except Exception as e:
        print(f"Image feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Debug: Extracting point features...")
    try:
        pts_feats = model.extract_pts_feat(points, img_metas)
        print("Point features extracted.")
        if pts_feats is None:
             print("pts_feats is None")
        elif isinstance(pts_feats, (list, tuple)):
             print("pts_feats shapes:", [f.shape for f in pts_feats])
        else:
             print("pts_feats shape:", pts_feats.shape)
    except Exception as e:
        print(f"Point feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    try:
        results = model.forward_test(img_metas, img=img, points=points)
        print("Forward pass successful!")
        if isinstance(results, tuple):
             print("Result type: tuple")
             print("Bev embed shape:", results[0].shape if hasattr(results[0], 'shape') else 'N/A')
             print("Bbox list length:", len(results[1]))
        else:
             print("Result type:", type(results))
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
