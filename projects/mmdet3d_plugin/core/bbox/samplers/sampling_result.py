
import jittor as jt

class SamplingResult(object):
    """Bbox sampling result."""

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = jt.empty(gt_bboxes.shape).to(bboxes.dtype)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return jt.concat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data instance."""
        # _t = self.pos_inds.to(device)
        # self.pos_inds = _t
        # self.neg_inds = self.neg_inds.to(device)
        # self.pos_bboxes = self.pos_bboxes.to(device)
        # self.neg_bboxes = self.neg_bboxes.to(device)
        # self.pos_is_gt = self.pos_is_gt.to(device)
        # self.pos_gt_bboxes = self.pos_gt_bboxes.to(device)
        return self

    def __repr__(self):
        return self.__class__.__name__ + f'(num_gts={self.num_gts}, ' \
            f'num_preds={len(self.pos_bboxes) + len(self.neg_bboxes)}, ' \
            f'num_pos={len(self.pos_bboxes)}, ' \
            f'num_neg={len(self.neg_bboxes)})'
