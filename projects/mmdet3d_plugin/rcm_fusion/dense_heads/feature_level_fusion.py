# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jisong Kim and Minjae Seong
# ---------------------------------------------

import copy
import jittor as jt
from jittor import nn
import numpy as np

from projects.mmdet3d_plugin.jittor_adapter import HEADS, build_bbox_coder, build_assigner, build_sampler, build_transformer, build_positional_encoding, FUSION_LAYERS, build_from_cfg
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

def multi_apply(func, *args, **kwargs):
    from functools import partial
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    x1 = x.clamp(eps, 1.0)
    x2 = (1 - x).clamp(eps, 1.0)
    return jt.log(x1/x2)

def reduce_mean(tensor):
    return tensor.mean()

# Dummy decorators
def auto_fp16(apply_to=None):
    def decorator(func):
        return func
    return decorator

def force_fp32(apply_to=None):
    def decorator(func):
        return func
    return decorator

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

# Placeholder for DETRHead
class DETRHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None):
        print(f"DEBUG: SigmoidFocalLoss execute. pred.shape={pred.shape}, target.shape={target.shape}")
        
        # Check NaNs in pred
        # if jt.isnan(pred).any():
        #     print("DEBUG: pred contains NaNs in SigmoidFocalLoss! replacing with 0")
        #     pred = jt.where(jt.isnan(pred), jt.zeros_like(pred), pred)
            
        pred_sigmoid = pred.sigmoid()
        
        # target_one_hot = nn.one_hot(target, num_classes=pred.shape[-1] + 1)[:, :pred.shape[-1]].float()
        # Safer one_hot implementation using scatter
        num_classes = pred.shape[-1]
        target_one_hot = jt.zeros((target.shape[0], num_classes + 1))
        # Ensure target is within range [0, num_classes]
        safe_target = jt.clamp(target, 0, num_classes)
        # Jittor scatter might not be inplace
        # Fix: scatter src must be a tensor, not int
        scatter_src = jt.ones(safe_target.unsqueeze(1).shape)
        target_one_hot = target_one_hot.scatter(1, safe_target.unsqueeze(1), scatter_src)
        target_one_hot = target_one_hot[:, :num_classes].float()
        
        print(f"DEBUG: target_one_hot.shape={target_one_hot.shape}")
        
        pt = (1 - pred_sigmoid) * target_one_hot + pred_sigmoid * (1 - target_one_hot)
        focal_weight = (self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)) * (pt.pow(self.gamma))
        print("DEBUG: focal_weight calculated")
        
        # Manual binary_cross_entropy_with_logits for element-wise loss
        # loss = nn.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none') * focal_weight
        # Stable BCE with logits
        max_val = jt.maximum(-pred, 0)
        bce_loss = (1 - target_one_hot) * pred + max_val + jt.log(jt.exp(-max_val) + jt.exp(-pred - max_val))
        
        print("DEBUG: bce_loss calculated")
        loss = bce_loss * focal_weight
        
        if weight is not None:
             if weight.ndim == 1:
                 weight = weight.unsqueeze(1)
             loss = loss * weight
             
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / avg_factor
        print("DEBUG: loss reduced")
        return loss * self.loss_weight

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
    
    def execute(self, pred, target, weight=None, avg_factor=None):
        loss = jt.abs(pred - target)
        if weight is not None:
            if weight.ndim == 1:
                weight = weight.unsqueeze(1)
            loss = loss * weight
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss * self.loss_weight

@HEADS.register_module()
class FeatureLevelFusion(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_classes=10, # Added explicit arg
                 embed_dims=256, # Added explicit arg
                 num_query=900, # Added explicit arg
                 num_reg_fcs=2, # Added explicit arg
                 loss_cls=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None,
                 pts_fusion_layer=None,
                 **kwargs):
        
        print(f"DEBUG: FeatureLevelFusion init. bev_h={bev_h}, bev_w={bev_w}, embed_dims={embed_dims}")
        super(FeatureLevelFusion, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        
        # Store args
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = num_classes # Assuming no background class or handled
        
        # Transformer setup (Assuming transformer is passed as object or built elsewhere)
        # In Jittor port, we might need to build it. For now assume it's passed or handle dict.
        if isinstance(transformer, dict):
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = transformer

        if pts_fusion_layer:
            self.pts_fusion_layer = build_from_cfg(pts_fusion_layer, FUSION_LAYERS)


        if self.as_two_stage:
            # transformer['as_two_stage'] = self.as_two_stage
            if hasattr(self.transformer, 'as_two_stage'):
                self.transformer.as_two_stage = self.as_two_stage

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
            
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        # BBox Coder
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        else:
            self.bbox_coder = None
        
        # self.pc_range = self.bbox_coder.pc_range
        # Mocking bbox_coder for now or extracting pc_range if in kwargs
        if self.bbox_coder is not None:
            self.pc_range = self.bbox_coder.pc_range
        else:
            self.pc_range = kwargs.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]) # Default
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        
        # self.code_weights = nn.Parameter(torch.tensor(
        #     self.code_weights, requires_grad=False), requires_grad=False)
        self.code_weights_var = jt.array(self.code_weights).stop_grad()

        # Initialize assigner and sampler
        if train_cfg:
            self.assigner = build_assigner(train_cfg['assigner'])
            # sampler is optional, default to PseudoSampler if not specified but config usually has it
            if 'sampler' in train_cfg:
                self.sampler = build_sampler(train_cfg['sampler'], context=self)
            else:
                # Default to PseudoSampler
                from projects.mmdet3d_plugin.core.bbox.samplers import PseudoSampler
                self.sampler = PseudoSampler()
        
        # Initialize losses
        if loss_cls:
            self.loss_cls = SigmoidFocalLoss(
                gamma=loss_cls.get('gamma', 2.0),
                alpha=loss_cls.get('alpha', 0.25),
                loss_weight=loss_cls.get('loss_weight', 1.0)
            )
        else:
            self.loss_cls = SigmoidFocalLoss()

        if loss_bbox:
            self.loss_bbox = L1Loss(
                loss_weight=loss_bbox.get('loss_weight', 1.0)
            )
        else:
            self.loss_bbox = L1Loss()
            
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = True # Default for Detr3D

        self._init_layers()
        
        # Positional encoding
        self.positional_encoding = build_positional_encoding(
            dict(type='SinePositionalEncoding', num_feats=self.embed_dims // 2, normalize=True))

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        # num_pred = (self.transformer.decoder.num_layers + 1) if \
        #     self.as_two_stage else self.transformer.decoder.num_layers
        
        # Assuming transformer has decoder.num_layers
        num_pred = 6 # Default
        if self.transformer and hasattr(self.transformer, 'decoder'):
             num_pred = (self.transformer.decoder.num_layers + 1) if \
                self.as_two_stage else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        # self.transformer.init_weights()
        pass 
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    def execute(self, mlvl_feats, img_metas, prev_bev=None, pts_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs = mlvl_feats[0].shape[0]
        # dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight
        bev_queries = self.bev_embedding.weight

        bev_mask = jt.zeros((bs, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                pts_bev=pts_bev,
                only_bev=only_bev
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                pts_bev=pts_bev,
                only_bev=only_bev
            )
        
        bev_embed, hs, init_reference, inter_references = outputs
        # hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl]) # cx, cy, w, l, cz, h, rots, rotc, vx, vy
            
            # TODO: check the shape of reference
            # assert reference.shape[-1] == 3
            if reference.shape[-1] == 2:
                reference_z = jt.zeros_like(reference[..., 0:1])
            else:
                reference_z = reference[..., 2:3]
            
            # Logic:
            # tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 4:5] += reference[..., 2:3]
            
            cx = tmp[..., 0:1] + reference[..., 0:1]
            cy = tmp[..., 1:2] + reference[..., 1:2]
            cz = tmp[..., 4:5] + reference_z

            cx = cx.sigmoid()
            cy = cy.sigmoid()
            cz = cz.sigmoid()
            
            # Scale
            cx = cx * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            cy = cy * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            cz = cz * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            # Re-reassemble
            # cx, cy, w, l, cz, h, ...
            # tmp has: cx, cy, w, l, cz, h, ...
            # We computed new cx, cy, cz
            # w, l, h, ... are in tmp[..., 2:4] and tmp[..., 5:]
            
            parts = [
                cx,
                cy,
                tmp[..., 2:4], # w, l
                cz,
                tmp[..., 5:] # h, ...
            ]
            tmp = jt.concat(parts, dim=-1)

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_classes = jt.stack(outputs_classes)
        outputs_coords = jt.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'head_features': hs
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        """
        num_bboxes = bbox_pred.shape[0]
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = jt.full((num_bboxes,),
                                    self.num_classes,
                                    dtype='int32')
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # Jittor indexing might be tricky if pos_inds is empty or tensor
        if pos_inds.numel() > 0:
             labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
             
        label_weights = jt.ones((num_bboxes,))

        # bbox targets
        bbox_targets = jt.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = jt.zeros_like(bbox_pred)
        
        if pos_inds.numel() > 0:
            bbox_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
            
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        """
        print("DEBUG: Entering loss_single")
        num_imgs = cls_scores.shape[0]
        
        # Check for NaNs
        if jt.isnan(cls_scores).any():
             print("DEBUG: cls_scores contains NaNs!")
        if jt.isnan(bbox_preds).any():
             print("DEBUG: bbox_preds contains NaNs!")
             
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        print("DEBUG: Calling get_targets")
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        print(f"DEBUG: labels_list len={len(labels_list)}")
        labels = jt.concat(labels_list, 0)
        label_weights = jt.concat(label_weights_list, 0)
        bbox_targets = jt.concat(bbox_targets_list, 0)
        bbox_weights = jt.concat(bbox_weights_list, 0)
        print(f"DEBUG: labels shape={labels.shape}")

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                jt.array([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        print("DEBUG: Calling loss_cls")
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        print("DEBUG: loss_cls returned")

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = jt.array([num_total_pos])
        # num_total_pos = jt.clamp(reduce_mean(num_total_pos), min_v=1).item()
        num_total_pos = jt.clamp(reduce_mean(num_total_pos), 1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.shape[-1])
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        
        # isnotnan check
        # Jittor doesn't have isfinite easily accessible on vars, let's assume valid data for now
        # or implement simple check: x==x
        # isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        isnotnan = (normalized_bbox_targets == normalized_bbox_targets).all(dim=-1)
        
        # bbox_weights = bbox_weights * self.code_weights
        bbox_weights = bbox_weights * self.code_weights_var
        
        # Jittor indexing with boolean mask might need nonzero()
        # bbox_preds[isnotnan]
        
        # In Jittor, boolean indexing works.
        
        print("DEBUG: Calling loss_bbox")
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        print("DEBUG: loss_bbox returned")
        
        return loss_cls, loss_bbox

    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        """Loss function."""
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        
        # device = gt_labels_list[0].device
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        # Assuming gt_bboxes_list are already tensors in correct format or need conversion
        # In Jittor port, we expect them to be Jittor arrays.
        
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map
        if enc_cls_scores is not None:
            binary_labels_list = [
                jt.zeros_like(gt_labels_list[i])
                for i in range(len(gt_labels_list))
            ]
            enc_loss_cls, enc_loss_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_loss_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from predictions."""
        if self.bbox_coder is None:
            return []
            
        # Save head_features before overwriting preds_dicts
        head_features = preds_dicts.get('head_features', None) # (num_dec, bs, num_query, C)
        
        decoded_preds = self.bbox_coder.decode(preds_dicts)
        
        num_samples = len(decoded_preds)
        ret_list = []
        for i in range(num_samples):
            preds = decoded_preds[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            
            # Handle head_features if available
            if head_features is not None:
                # Use the last decoder layer features: (bs, num_query, C)
                last_layer_head_features = head_features[-1] 
                
                # Get indices from coder output to select corresponding features
                if 'indices' in preds:
                    indices = preds['indices']
                    # Select features: (num_selected, C)
                    # Jittor indexing
                    selected_head_features = last_layer_head_features[i][indices]
                else:
                    # Fallback if no indices (e.g. legacy coder), return all or None?
                    # If we don't have indices, we can't map boxes to features correctly 
                    # unless coder didn't filter/reorder. 
                    # Assuming NMSFreeCoder with topk, we need indices.
                    # For now, just return raw features if indices missing (might be wrong shape)
                    selected_head_features = last_layer_head_features[i]
                
                ret_list.append([bboxes, selected_head_features, scores, labels])
            else:
                ret_list.append([bboxes, scores, labels])
                
        return ret_list
