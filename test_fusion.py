print("DEBUG: Starting test_fusion.py")
with open("checkpoint_start.txt", "w") as f: f.write("START")
import os
# Force Jittor to use serial compilation to avoid OOM/Crash on macOS
os.environ["nvcc_threads"] = "1" 
os.environ["cpu_threads"] = "1"

import jittor as jt
# Disable parallel op compiler
jt.flags.use_parallel_op_compiler = 0

print("DEBUG: Imported jittor")
with open("checkpoint_import.txt", "w") as f: f.write("IMPORT")
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from projects.mmdet3d_plugin.rcm_fusion.dense_heads.feature_level_fusion import FeatureLevelFusion
from projects.mmdet3d_plugin.jittor_adapter import TRANSFORMER, TRANSFORMER_LAYER

def test_feature_level_fusion():
    print("Testing FeatureLevelFusion...")
    print(f"test_fusion.py TRANSFORMER_LAYER id: {id(TRANSFORMER_LAYER)}")
    print(f"TRANSFORMER_LAYER registry: {TRANSFORMER_LAYER.module_dict.keys()}")
    
    # Mock config
    embed_dims = 256
    num_classes = 10
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # We need to ensure all classes are imported so registries are populated
    # Imports for ported components
    from projects.mmdet3d_plugin.rcm_fusion.modules import radar_guided_bev_encoder
    from projects.mmdet3d_plugin.rcm_fusion.modules import radar_guided_bev_attention
    from projects.mmdet3d_plugin.rcm_fusion.modules import spatial_cross_attention
    from projects.mmdet3d_plugin.rcm_fusion.modules import decoder
    from projects.mmdet3d_plugin.rcm_fusion.modules import transformer_radar
    from projects.mmdet3d_plugin.core.bbox.coders import nms_free_coder
    from projects.mmdet3d_plugin.core.bbox.assigners import hungarian_assigner_3d
    from projects.mmdet3d_plugin.core.bbox.match_costs import match_cost
    from projects.mmdet3d_plugin.core.bbox.samplers import pseudo_sampler

    head = FeatureLevelFusion(
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=100,
        transformer=dict(
            type='PerceptionTransformerRadar',
            embed_dims=embed_dims,
            num_feature_levels=4,
            num_cams=6,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            can_bus_norm=True,
            use_cams_embeds=True,
            rotate_center=[100, 100],
            encoder=dict(
                type='RadarGuidedBEVEncoder',
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='RadarGuidedBEVEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='RadarGuidedBEVAttention',
                            embed_dims=embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            embed_dims=embed_dims,
                            num_cams=6,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=embed_dims,
                                num_levels=4,
                                num_points=8))
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    batch_first=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            dropout=0.1,
                            batch_first=False),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_levels=1,
                            batch_first=False)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=pc_range,
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=num_classes),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='SigmoidFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=pc_range),
            sampler=dict(type='PseudoSampler')
        )
    )

    print("Head initialized successfully.")
    
    # Create dummy inputs
    bs = 1
    num_cams = 6
    # mlvl_feats shapes: (bs, num_cams, C, H, W)
    # H, W decrease by 2 each level
    # Use numpy to avoid potential Jittor randn issues
    mlvl_feats = [
        jt.array(np.random.randn(bs, num_cams, embed_dims, 20, 20).astype(np.float32)),
        jt.array(np.random.randn(bs, num_cams, embed_dims, 10, 10).astype(np.float32)),
        jt.array(np.random.randn(bs, num_cams, embed_dims, 5, 5).astype(np.float32)),
        jt.array(np.random.randn(bs, num_cams, embed_dims, 3, 3).astype(np.float32)), # simplified
    ]
    
    img_metas = [{
        'can_bus': np.zeros(18),
        'lidar2img': [np.eye(4) for _ in range(num_cams)],
        'img_shape': [(900, 1600, 3) for _ in range(num_cams)],
        'box_type_3d': None 
    }]
    
    # Create dummy pts_bev
    # pts_bev shape should be (H*W, bs, embed_dims) to match bev_query in RadarGuidedBEVEncoder
    print("DEBUG: Creating pts_bev using numpy...")
    pts_bev_np = np.random.randn(30*30, 1, 256).astype(np.float32)
    pts_bev = jt.array(pts_bev_np)

    jt.sync_all(True)
    if jt.isnan(pts_bev).any():
        print("DEBUG: pts_bev contains NaNs immediately after creation!")

    # Forward
    print("Running forward...")
    with open("checkpoint_init.txt", "w") as f: f.write("INIT_DONE")
    # FeatureLevelFusion.forward(self, mlvl_feats, img_metas)
    outs = head(mlvl_feats, img_metas, pts_bev=pts_bev)
    with open("checkpoint_forward.txt", "w") as f: f.write("FORWARD_DONE")
    print("Forward successful.")
    
    # Check for NaNs in outs
    if 'all_cls_scores' in outs:
        if jt.isnan(outs['all_cls_scores']).any():
             print("ERROR: all_cls_scores contains NaNs!")
             # Print some values
             print(outs['all_cls_scores'][0, :5, :5])
    if 'all_bbox_preds' in outs:
        if jt.isnan(outs['all_bbox_preds']).any():
             print("ERROR: all_bbox_preds contains NaNs!")
             print(outs['all_bbox_preds'][0, :5, :5])
    
    # Loss
    print("Running loss...")
    with open("checkpoint_preloss.txt", "w") as f: f.write("PRE_LOSS")
    # gt_bboxes_list: list of tensors (num_gts, 9)
    # gt_labels_list: list of tensors (num_gts)
    gt_bboxes_list = [jt.array(np.random.randn(5, 9).astype(np.float32))] 
    gt_labels_list = [jt.randint(0, num_classes, (5,))]
    
    # preds_dicts = outs
    loss_dict = head.loss(gt_bboxes_list, gt_labels_list, outs)
    with open("checkpoint_loss_done.txt", "w") as f: f.write("LOSS_DONE")
    print("Loss successful.")
    print(loss_dict)

if __name__ == "__main__":
    test_feature_level_fusion()
