
import jittor as jt
from .sampling_result import SamplingResult
from projects.mmdet3d_plugin.jittor_adapter import BBOX_SAMPLERS

class BaseSampler(object):
    """Base class of samplers."""

    def __init__(self,
                 num=None,
                 pos_fraction=None,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes."""
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype='uint8')
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = jt.concat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype='uint8')
            gt_flags = jt.concat([gt_ones, gt_flags])

        num_expected = self.num
        pos_inds = self._sample_pos(assign_result, num_expected, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = num_expected - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(assign_result, num_expected_neg, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive bboxes."""
        raise NotImplementedError

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative bboxes."""
        raise NotImplementedError


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling in fact."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive bboxes."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative bboxes."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (torch.Tensor): Boxes to be sampled.
            gt_bboxes (torch.Tensor): Ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        pos_inds = jt.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = jt.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = jt.zeros(bboxes.shape[0], dtype='uint8')
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
