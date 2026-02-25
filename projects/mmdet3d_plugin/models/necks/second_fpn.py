import jittor as jt
from jittor import nn
import numpy as np

from projects.mmdet3d_plugin.jittor_adapter import NECKS

# Placeholder for BaseModule
class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

# @NECKS.register_module()
# class SECONDFPN_v2(BaseModule):
#     """FPN used in SECOND/PointPillars/PartA2/MVXNet.
#     
#     Note: Porting to Jittor requires removing mmcv dependencies.
#     This class is currently disabled until fully ported.
#     """
#     def __init__(self,
#                  in_channels=[128, 128, 256],
#                  out_channels=[256, 256, 256],
#                  upsample_strides=[1, 2, 4],
#                  fused_channels_in=None,
#                  fused_channels_out=None,
#                  norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
#                  upsample_cfg=dict(type='deconv', bias=False),
#                  conv_cfg=dict(type='Conv2d', bias=False),
#                  use_conv_for_no_stride=False,
#                  init_cfg=None):
#         super(SECONDFPN_v2, self).__init__(init_cfg=init_cfg)
#         pass

@NECKS.register_module()
class BaseBEVBackboneV1(nn.Module):
    def __init__(self, layer_nums,  num_filters, upsample_strides, num_upsample_filters):
        super().__init__()
        assert len(layer_nums) == len(num_filters) == 2
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1), # ZeroPad2d(1)
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = int(np.round(1 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def execute(self, data_dict, bs):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # Add debug prints
        print(f"DEBUG: BaseBEVBackboneV1 execute start. bs: {bs}")
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        x_0 = self.blocks[0](x_conv4)
        ups = [self.deblocks[0](x_0)]

        x_1 = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x_1))

        # Fix shape mismatch if any (e.g. 25x25 vs 50x50)
        if len(ups) > 1:
            target_h, target_w = ups[0].shape[2], ups[0].shape[3]
            for i in range(1, len(ups)):
                h, w = ups[i].shape[2], ups[i].shape[3]
                if h != target_h or w != target_w:
                    # Resize to target size
                    ups[i] = nn.interpolate(ups[i], size=(target_h, target_w), mode='bilinear', align_corners=False)

        x = jt.concat(ups, dim=1)

        return x.view(bs, 256, -1).permute(2, 0, 1)


@NECKS.register_module()
class BaseBEVBackboneV2(nn.Module):
    def __init__(self, layer_nums,  num_filters, upsample_strides, num_upsample_filters):
        super().__init__()
        assert len(layer_nums) == len(num_filters) == 2
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            if idx == 0:
                cur_layers = [
                    nn.ZeroPad2d(1), # ZeroPad2d(1)
                    nn.Conv2d(num_filters[idx]*2, num_filters[idx], kernel_size=3,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            else:
                cur_layers = [
                    nn.ZeroPad2d(1), # ZeroPad2d(1)
                    nn.Conv2d(
                        num_filters[idx], num_filters[idx], kernel_size=3,
                        stride=1, padding=0, bias=False
                    ),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose(
                            num_filters[idx], num_upsample_filters[idx]*2,
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx]*2, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = int(np.round(1 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.deblocks = self.deblocks[1:]

    def execute(self, data_dict, bs):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [x_conv4]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))

        x = jt.concat(ups, dim=1)
        x = self.blocks[0](x)

        return x.view(bs, 256, -1).permute(2, 0, 1)
