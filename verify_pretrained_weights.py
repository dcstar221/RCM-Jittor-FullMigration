
import sys
import os
import os
# Set Jittor env vars before import
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['use_mkl'] = '0'
import jittor as jt
import numpy as np
import pickle

# Add project root to path
sys.path.append(os.getcwd())

from projects.mmdet3d_plugin.jittor_adapter import build_detector
import projects.mmdet3d_plugin.rcm_fusion.detectors.rcm_fusion_jittor
import projects.mmdet3d_plugin.models.backbones.spconv_backbone_2d
import projects.mmdet3d_plugin.models.backbones.resnet_jittor
import projects.mmdet3d_plugin.models.necks.fpn_jittor
import projects.mmdet3d_plugin.models.necks.second_fpn
import projects.mmdet3d_plugin.rcm_fusion.dense_heads.feature_level_fusion
import projects.mmdet3d_plugin.models.fusion_layers.instance_level_fusion_jittor
import projects.mmdet3d_plugin.rcm_fusion.modules.radar_guided_bev_encoder
import projects.mmdet3d_plugin.rcm_fusion.modules.transformer_radar
import projects.mmdet3d_plugin.rcm_fusion.modules.radar_guided_bev_attention
import projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder
import projects.mmdet3d_plugin.models.vfe.dynamic_pillar_vfe
import projects.mmdet3d_plugin.rcm_fusion.modules.custom_base_transformer_layer
import projects.mmdet3d_plugin.rcm_fusion.modules.decoder
import projects.mmdet3d_plugin.rcm_fusion.modules.detr3d_cross_attention_jittor
import projects.mmdet3d_plugin.rcm_fusion.modules

def load_weights(model, weight_path):
    print(f"Loading weights from {weight_path}...")
    with open(weight_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    model_state = model.state_dict()
    matched = 0
    mismatched = 0
    missing = 0
    extra = 0
    
    # Check strict match first
    model_keys = set(model_state.keys())
    loaded_keys = set(state_dict.keys())
    
    # Jittor keys often don't have 'module.' prefix, PyTorch might not either if saved from model
    # But sometimes PyTorch has 'module.' if DataParallel was used.
    # Check if we need to strip 'module.'
    first_key = list(loaded_keys)[0]
    if first_key.startswith('module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
        loaded_keys = set(state_dict.keys())
        print("Stripped 'module.' prefix from loaded keys.")

    for k, v in state_dict.items():
        if k in model_state:
            try:
                # Check shape
                target_shape = model_state[k].shape
                source_shape = v.shape
                
                # Handle possible shape mismatch due to batch_first or other changes
                if target_shape != source_shape:
                    # Transpose 2D weights if they are flipped (e.g. Linear layers in some frameworks)
                    # But Jittor and PyTorch usually match for Linear (out, in).
                    # Maybe Conv2d? (out, in, h, w).
                    # Check if transpose fixes it
                    if len(target_shape) == 2 and len(source_shape) == 2 and \
                       target_shape[0] == source_shape[1] and target_shape[1] == source_shape[0]:
                        v = v.T
                        print(f"Transposed {k} to match shape.")
                    else:
                        print(f"Shape mismatch for {k}: Model {target_shape} vs Loaded {source_shape}")
                        mismatched += 1
                        continue
                
                model_state[k].assign(v)
                matched += 1
            except Exception as e:
                print(f"Error loading {k}: {e}")
                mismatched += 1
        else:
            # print(f"Extra key in loaded weights: {k}")
            extra += 1
            
    for k in model_state:
        if k not in state_dict:
            # print(f"Missing key in loaded weights: {k}")
            missing += 1
            
    print(f"Summary: Matched: {matched}, Mismatched: {mismatched}, Missing: {missing}, Extra: {extra}")
    return matched > 0

def main():
    print("Starting verification with pretrained weights...")
    jt.flags.use_cuda = 0 
    
    # Config matching ResNet-50 and Pretrained parameters
    # Key params: bev_h=200, bev_w=200, embed_dims=256
    
    voxel_size = [0.512, 0.512, 0.2] # 102.4 / 200 = 0.512
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    grid_size = [200, 200, 40] 
    bev_h = 200
    bev_w = 200
    
    model_cfg = dict(
        type='RCMFusionJittor',
        freeze_img=False,
        freeze_pts=False,
        use_grid_mask=True,
        pts_voxel_encoder=dict(
            type='DynamicPillarVFESimple2D',
            num_point_features=6,
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
            layer_nums=[1, 1],
            num_filters=[256, 256],
            upsample_strides=[8, 16],
            num_upsample_filters=[128, 128]),
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            with_cp=True),
        img_neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=4,
            relu_before_extra_convs=True),
        pts_bbox_head=dict(
            type='FeatureLevelFusion',
            bev_h=bev_h,
            bev_w=bev_w,
            num_query=900,
            num_classes=10,
            in_channels=256,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            bbox_coder=dict(
                type='NMSFreeCoder',
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                voxel_size=voxel_size,
                num_classes=10,
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            transformer=dict(
                type='PerceptionTransformerRadar',
                embed_dims=256,
                num_feature_levels=4,
                num_cams=6, # Reverted to 6
                encoder=dict(
                    type='RadarGuidedBEVEncoder',
                    num_layers=1,
                    pc_range=point_cloud_range,
                    voxel_size=voxel_size,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='RadarGuidedBEVEncoderLayer',
                        attn_cfgs=[
                            dict(
                                type='RadarGuidedBEVAttention',
                                embed_dims=256,
                                num_levels=1,
                                num_points=4,
                            ),
                            dict(
                                type='RadarGuidedBEVAttention',
                                embed_dims=256,
                                num_levels=4,
                                num_points=4,
                            )
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
                    ),
                ),
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=256,
                                num_heads=8,
                                num_levels=1,
                                num_points=4,
                            ),
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
                    ),
                ),
            ),
            pts_fusion_layer=dict(
                type='InstanceLevelFusion',
                input_channels=64,
                code_size=7,
                sa_mlps=[[128, 128], [128, 128]],
                grid_fps=64,
                radii=[0.4, 0.8],
                num_samples=[16, 32],
                dilated_group=[False, False],
                norm_cfg=dict(type='BN2d'),
                sa_cfg=dict(
                    type='PointNetSA',
                    pool_mod='max',
                    use_xyz=True,
                    normalize_xyz=True
                ),
                grid_size=[6, 6, 1],
                num_classes=10,
            ),
        )
    )

    print("Building model...")
    model = build_detector(model_cfg)
    
    # Debug: Check InstanceLevelFusion structure
    if hasattr(model.pts_bbox_head, 'pts_fusion_layer'):
        print("\n--- Debug InstanceLevelFusion ---")
        fusion = model.pts_bbox_head.pts_fusion_layer
        print(f"Fusion Layer Type: {type(fusion)}")
        if hasattr(fusion, 'reg_layer'):
            print(f"reg_layer: {fusion.reg_layer}")
            if hasattr(fusion.reg_layer, 'weight'):
                 print(f"reg_layer.weight.shape: {fusion.reg_layer.weight.shape}")
        if hasattr(fusion, 'shared_fc_layer'):
             print(f"shared_fc_layer: {fusion.shared_fc_layer}")
        print("---------------------------------\n")

    weight_path = "/Users/dishangpeng/Desktop/RCMmacosjittor/pretrain/rcm-fusion-r50-icra-final-numpy.pkl"
    if load_weights(model, weight_path):
        print("Weights loaded successfully!")
        
        # Run forward pass
        print("Running forward pass...")
        
        # Mock Data
        B = 1
        img = jt.randn((B, 6, 3, 480, 800)) # reduced size
        points = [jt.randn((1000, 6))]
        img_metas = [dict(
            box_type_3d=None,
            lidar2img=[np.eye(4) for _ in range(6)],
            img_shape=[(480, 800, 3)] * 6,
            can_bus=np.zeros(18)
        )]
        radar_points = [jt.randn((100, 7))] # xyz, vel_x, vel_y, rcs, dummy
        
        try:
            with jt.no_grad():
                # Use return_loss=False to trigger simple_test
                outputs = model(
                    points=points,
                    img_metas=img_metas,
                    img=img,
                    return_loss=False,
                    rescale=True
                )
            print("Forward pass successful!")
            print(f"Output count: {len(outputs)}")
            # print(outputs)
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Weight loading failed significantly.")

if __name__ == "__main__":
    main()
