import jittor as jt
from jittor import nn
from projects.mmdet3d_plugin.jittor_adapter import MATCH_COST

@MATCH_COST.register_module()
class BBoxL1Cost(object):
    """BBoxL1Cost.
     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
     """
    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Var): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Var): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # For Jittor, use jt.abs and sum
        bbox_cost = jt.abs(bbox_pred.unsqueeze(1) - gt_bboxes.unsqueeze(0)).sum(-1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class ClassificationCost(object):
    """ClsSoftmaxCost.
     Args:
         weight (int | float, optional): loss_weight
     """
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Var): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Var): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Usually uses sigmoid focal loss style cost
        # cls_pred is logits
        # We need to compute negative log likelihood or similar
        # Standard DETR uses: -alpha * (1-p)^gamma * log(p) for positive class
        # But here let's assume standard implementation
        # For simplicity, let's implement the one used in mmdet
        
        # This is a simplified version often used in Hungarian matching
        # cost = - prob[gt_label]
        
        cls_score = cls_pred.sigmoid()
        # Create a matrix of probabilities for each gt label
        # shape: [num_query, num_gt]
        
        # We want P(class = gt_label) for each query and each gt
        # cls_score: [num_query, num_classes]
        # gt_labels: [num_gt]
        
        # Gather the probabilities corresponding to gt_labels
        # We need to expand cls_score or index into it
        
        # Efficient way:
        # cls_score[:, gt_labels] gives [num_query, num_gt]
        # But we need to be careful with indices
        
        cls_cost = -cls_score[:, gt_labels]
        
        return cls_cost * self.weight

@MATCH_COST.register_module()
class IoUCost(object):
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss_weight
     """
    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Var): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Var): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # Placeholder for IoU calculation
        # Implementing full GIoU in Jittor requires some work
        # For now, return zeros if weight is 0, or implement simple IoU
        if self.weight == 0:
            return jt.zeros((bbox_pred.shape[0], gt_bboxes.shape[0]))
            
        # TODO: Implement IoU calculation
        return jt.zeros((bbox_pred.shape[0], gt_bboxes.shape[0]))
