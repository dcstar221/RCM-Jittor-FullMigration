from functools import partial

import jittor as jt
import jittor.nn as nn
import numpy as np

from projects.mmdet3d_plugin.jittor_adapter import BACKBONES

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    # Jittor Dense Implementation
    # conv_type is ignored (treated as dense conv)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    m = nn.Sequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        # SubMConv2d becomes Conv2d with stride=1, padding=1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class SparseBasicBlockV(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlockV, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv0 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn0 = norm_fn(planes)
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class PillarRes18BackBone8x(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01) # Use 2D Batchnorm
        self.sparse_shape = np.array(grid_size)[[1,0]]
        self.grid_size = grid_size
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = nn.Sequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def execute(self, pillar_features, pillar_coords, batch_size):
        # Scatter to Dense
        # pillar_coords: [M, 4] (batch_idx, z, y, x)
        # grid_size: [W, H] (from config, but usually [H, W] or [X, Y]?)
        # self.sparse_shape = grid_size[[1,0]] which implies grid_size is [W, H] and shape is [H, W]
        # Let's assume sparse_shape is [H, W]
        
        H, W = int(self.sparse_shape[0]), int(self.sparse_shape[1])
        C = pillar_features.shape[1]
        
        # Create canvas
        canvas = jt.zeros((batch_size, C, H, W), dtype=pillar_features.dtype)
        
        # Extract indices
        # pillar_coords is (batch_idx, z, y, x)
        # We need (batch_idx, C, y, x)
        # Since we want to fill all channels C with the feature vector, we use index broadcasting or loop?
        # Jittor supports advanced indexing.
        
        b_idx = pillar_coords[:, 0].int()
        y_idx = pillar_coords[:, 2].int()
        x_idx = pillar_coords[:, 3].int()
        
        # Canvas index: [b, :, y, x] = feature
        # We can flatten canvas to [B*H*W, C]? No.
        # Use scatter?
        # canvas[b, :, y, x] = features
        # This is hard in Jittor without loop if C is large?
        # Actually, we can just use fancy indexing if Jittor supports it fully.
        # canvas[b_idx, :, y_idx, x_idx] = pillar_features -> this might not work if Jittor expects index for ':'
        
        # Alternative: ScatterND or similar.
        # Or flatten indices: idx = b * H * W + y * W + x
        # canvas_flat = canvas.view(-1, C)
        # flat_idx = b_idx * H * W + y_idx * W + x_idx
        # canvas_flat[flat_idx] = pillar_features
        # canvas = canvas_flat.view(batch_size, C, H, W)
        
        flat_idx = b_idx * H * W + y_idx * W + x_idx
        canvas_flat = canvas.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
        
        # We need to assign.
        # Note: If multiple pillars map to same voxel (shouldn't happen with unique), last one wins or sum?
        # VFE should produce unique voxels.
        
        # Jittor indexing:
        # canvas_flat[flat_idx] = pillar_features
        # This works in Jittor.
        
        canvas_flat[flat_idx] = pillar_features
        x = canvas_flat.reshape(batch_size, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]

        spconv_backbone_outs = dict()
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        # x_conv4 is already dense
        x_conv5 = self.conv5(x_conv4)

        spconv_backbone_outs.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        spconv_backbone_outs.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return spconv_backbone_outs


@BACKBONES.register_module()
class PillarRes18BackBone8x2(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01) # Use 2D Batchnorm
        self.sparse_shape = np.array(grid_size)[[1,0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = nn.Sequential(
            SparseBasicBlockV(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlockV(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlockV(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlockV(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlockV(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlockV(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlockV(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlockV(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def execute(self, pillar_features, pillar_coords, batch_size):
        print(f"DEBUG: PillarRes18BackBone8x2 execute start. pillar_features={pillar_features.shape}")
        # Scatter to Dense
        H, W = int(self.sparse_shape[0]), int(self.sparse_shape[1])
        C = pillar_features.shape[1]
        
        # Create canvas
        canvas = jt.zeros((batch_size, C, H, W), dtype=pillar_features.dtype)
        
        b_idx = pillar_coords[:, 0].int()
        y_idx = pillar_coords[:, 2].int()
        x_idx = pillar_coords[:, 3].int()
        
        flat_idx = b_idx * H * W + y_idx * W + x_idx
        canvas_flat = canvas.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
        
        canvas_flat[flat_idx] = pillar_features
        x = canvas_flat.reshape(batch_size, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]
        
        spconv_backbone_outs = dict()
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        spconv_backbone_outs.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        spconv_backbone_outs.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return spconv_backbone_outs