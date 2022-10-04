import torch
import torch.nn.functional as F

try:
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.dense_heads import YOLOXHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class YOLOXHeadAVA(YOLOXHead):
        @torch.no_grad()
        def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                               gt_bboxes, gt_labels):
            """Compute classification, regression, and objectness targets for
            priors in a single image.
            Args:
                cls_preds (Tensor): Classification predictions of one image,
                    a 2D-Tensor with shape [num_priors, num_classes]
                objectness (Tensor): Objectness predictions of one image,
                    a 1D-Tensor with shape [num_priors]
                priors (Tensor): All priors of one image, a 2D-Tensor with shape
                    [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
                decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                    a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                    br_x, br_y] format.
                gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                    with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (Tensor): Ground truth labels of one image, a Tensor
                    with shape [num_gts].
            """

            num_priors = priors.size(0)
            num_gts = gt_labels.size(0)
            gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
            # No target
            if num_gts == 0:
                cls_target = cls_preds.new_zeros((0, self.num_classes))
                bbox_target = cls_preds.new_zeros((0, 4))
                l1_target = cls_preds.new_zeros((0, 4))
                obj_target = cls_preds.new_zeros((num_priors, 1))
                foreground_mask = cls_preds.new_zeros(num_priors).bool()
                return (foreground_mask, cls_target, obj_target, bbox_target,
                        l1_target, 0)

            # YOLOX uses center priors with 0.5 offset to assign targets,
            # but use center priors without offset to regress bboxes.
            offset_priors = torch.cat(
                [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

            assign_result = self.assigner.assign(
                cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
                offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

            sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            num_pos_per_img = pos_inds.size(0)

            pos_ious = assign_result.max_overlaps[pos_inds]
            # IOU aware classification score
            if len(pos_ious) == 0:
                cls_target = F.one_hot(sampling_result.pos_gt_labels,
                                       self.num_classes) * pos_ious.unsqueeze(-1)
            else:
                cls_target = sampling_result.pos_gt_labels * pos_ious.unsqueeze(-1)
            obj_target = torch.zeros_like(objectness).unsqueeze(-1)
            obj_target[pos_inds] = 1
            bbox_target = sampling_result.pos_gt_bboxes
            l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
            if self.use_l1:
                l1_target = self._get_l1_target(l1_target, bbox_target,
                                                priors[pos_inds])
            foreground_mask = torch.zeros_like(objectness).to(torch.bool)
            foreground_mask[pos_inds] = 1
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, num_pos_per_img)

else:
    # Just define an empty class, so that __init__ can import it.
    class YOLOXHeadAVA:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `HEADS` from `mmdet.models`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`YOLOXHeadAVA`. ')
