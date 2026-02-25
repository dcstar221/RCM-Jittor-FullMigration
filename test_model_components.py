
import jittor as jt
import numpy as np
from projects.mmdet3d_plugin.rcm_fusion.dense_heads.feature_level_fusion import FeatureLevelFusion
from projects.mmdet3d_plugin.jittor_adapter import build_transformer

def test_feature_level_fusion():
    print("Testing FeatureLevelFusion...")
    
    # Configuration based on rcm-fusion_r101.py
    _dim_ = 256
    _num_levels_ = 4
    bev_h_ = 50 # Reduced for testing
    bev_w_ = 50
    
    transformer_config = dict(
        type='PerceptionTransformerRadar',
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        embed_dims=_dim_,
        encoder=dict(
            type='RadarGuidedBEVEncoder',
            num_layers=1, # Reduced
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='RadarGuidedBEVEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='RadarGuidedBEVAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_),
                        embed_dims=_dim_,
                    ),
                ],
                feedforward_channels=_dim_ * 2,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        decoder=dict(
            type='DetectionTransformerDecoder',
            num_layers=1, # Reduced
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                    dict(
                        type='CustomMSDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                ],
                feedforward_channels=_dim_ * 2,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))))
    
    head = FeatureLevelFusion(
        with_box_refine=True,
        as_two_stage=False,
        transformer=transformer_config,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=10,
        embed_dims=_dim_,
        num_query=100 # Reduced
    )
    
    # Initialize weights (mock)
    head.init_weights()
    
    # Create dummy inputs
    bs = 2  # Restored to 2 to verify segfault fix
    num_heads = 8
    embed_dims = 256
    num_queries = 2500 # 50*50
    num_points = 4
    num_levels = 1
    mlvl_feats = []
    feat_shapes = [(50, 50), (25, 25), (13, 13), (7, 7)]
    for lvl in range(4):
        h, w = feat_shapes[lvl]
        feat = jt.randn(bs, 6, 256, h, w)
        mlvl_feats.append(feat)

    img_metas = []
    for _ in range(bs):
        img_metas.append(dict(
            lidar2img=[np.eye(4) for _ in range(6)],
            img_shape=[(900, 1600) for _ in range(6)],
            can_bus=np.zeros(18),
            scene_token='dummy_scene',
            prev_bev=False,
            box_type_3d=None
        ))

    # Points in BEV (batch, embed_dims, H, W)
    # The head expects pts_bev to be (B, C, H, W)
    pts_bev = jt.randn(bs, 256, 50, 50)
    
    print("Executing head...")
    outputs = head.execute(mlvl_feats, img_metas, pts_bev=pts_bev)
    
    print("Execution successful!")
    
    if isinstance(outputs, dict):
        cls_scores = outputs['all_cls_scores']
        bbox_preds = outputs['all_bbox_preds']
        bev_embed = outputs['bev_embed']
        print(f"Output cls_scores shape: {cls_scores.shape}")
        print(f"Output bbox_preds shape: {bbox_preds.shape}")
        print(f"Output bev_embed shape: {bev_embed.shape}")
    else:
        # Fallback for old behavior
        cls_scores, bbox_preds = outputs
        print(f"cls_scores shape: {cls_scores.shape}") # Expected: (num_dec, bs, num_query, num_classes)
        print(f"bbox_preds shape: {bbox_preds.shape}") # Expected: (num_dec, bs, num_query, 10)
    
    # Simple check
    assert cls_scores.shape[-1] == 10
    assert bbox_preds.shape[-1] == 10
    
if __name__ == "__main__":
    try:
        test_feature_level_fusion()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
