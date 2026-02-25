
import jittor as jt
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from projects.mmdet3d_plugin.rcm_fusion.modules.radar_guided_bev_encoder import RadarGuidedBEVEncoder
from projects.mmdet3d_plugin.rcm_fusion.modules.transformer_radar import PerceptionTransformerRadar
from projects.mmdet3d_plugin.rcm_fusion.dense_heads.feature_level_fusion import FeatureLevelFusion
from projects.mmdet3d_plugin.models.necks.second_fpn import BaseBEVBackboneV2

def test_integration():
    print("Testing integration...")
    jt.flags.use_cuda = 0

    # 1. Setup Configs
    bev_h = 50
    bev_w = 50
    dim = 256
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # 2. Instantiate Components
    # Neck
    neck = BaseBEVBackboneV2(
        layer_nums=[3, 3],
        num_filters=[64, 128],
        upsample_strides=[1, 2],
        num_upsample_filters=[128, 128]
    )
    
    # Encoder Config
    encoder_cfg = dict(
        type='RadarGuidedBEVEncoder',
        num_layers=1,
        pc_range=pc_range,
        transformerlayers=dict(
            type='RadarGuidedBEVEncoderLayer',
            attn_cfgs=[
                dict(
                    type='RadarGuidedBEVAttention',
                    embed_dims=dim,
                    num_levels=1),
                dict(
                    type='SpatialCrossAttention',
                    pc_range=pc_range,
                    deformable_attention=dict(
                        type='MSDeformableAttention3D',
                        embed_dims=dim,
                        num_points=8,
                        num_levels=1),
                    embed_dims=dim)
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=dim,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    )

    # Decoder Config (Minimal for testing)
    decoder_cfg = dict(
        type='DetectionTransformerDecoder',
        num_layers=1,
        return_intermediate=True,
        transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                batch_first=False,
                attn_cfgs=[
                    dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
                dict(
                    type='CustomMSDeformableAttention',
                    embed_dims=dim,
                    num_levels=1)
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=dim,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    )

    # Transformer
    transformer = PerceptionTransformerRadar(
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        embed_dims=dim,
        num_feature_levels=1,
        num_cams=6
    )

    # Head
    head = FeatureLevelFusion(
        transformer=transformer,
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=dim,
        num_classes=10,
        num_query=900, # Matches decoder query count usually, or handled internally
        code_size=10
    )
    
    print("Testing FeatureLevelFusion head execution...")
    
    bs = 1
    num_query = 900
    
    # Inputs
    # mlvl_feats should be (bs, num_cams, dim, h, w)
    num_cams = 6
    mlvl_feats = [jt.randn((bs, num_cams, dim, 20, 20))] # Single scale
    
    # img_metas
    # lidar2img should be (num_cams, 4, 4) for each sample
    lidar2img = np.stack([np.eye(4) for _ in range(num_cams)])
    img_metas = [{'lidar2img': lidar2img, 'img_shape': [(900, 1600)] * num_cams, 'can_bus': np.zeros(18)}]
    
    pts_bev = jt.randn((bs, dim, bev_h, bev_w)) 
    pts_bev = pts_bev.flatten(2).permute(2, 0, 1) # (H*W, bs, dim)
    
    try:
        # Head execute
        # execute(self, mlvl_feats, img_metas, prev_bev=None, pts_bev=None, only_bev=False)
        outputs = head.execute(
            mlvl_feats=mlvl_feats,
            img_metas=img_metas,
            pts_bev=pts_bev,
            only_bev=False
        )
        
        # outputs is a dict
        print("Head execution successful!")
        print(f"Output keys: {outputs.keys()}")
        
        cls_scores = outputs['all_cls_scores']
        bbox_preds = outputs['all_bbox_preds']
        
        print(f"cls_scores shape: {cls_scores.shape}")
        print(f"bbox_preds shape: {bbox_preds.shape}")
        
    except Exception as e:
        print(f"Head failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
