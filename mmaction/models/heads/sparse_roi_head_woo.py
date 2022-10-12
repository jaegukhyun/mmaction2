import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
    from mmdet.core.bbox.samplers import PseudoSampler
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models import ROI_EXTRACTORS as MMDET_ROI
    from mmdet.models.roi_heads import SparseRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class SparseRoIHeadWOO(SparseRoIHead):
        def forward_train(self,
                          x,
                          proposal_boxes,
                          proposal_features,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          imgs_whwh=None,
                          gt_masks=None):
            """Forward function in training stage.

            Args:
                x (list[Tensor]): list of multi-level img features.
                proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                    (batch_size, num_proposals, 4)
                proposal_features (Tensor): Expanded proposal
                    features, has shape
                    (batch_size, num_proposals, proposal_feature_channel)
                img_metas (list[dict]): list of image info dict where
                    each dict has: 'img_shape', 'scale_factor', 'flip',
                    and may also contain 'filename', 'ori_shape',
                    'pad_shape', and 'img_norm_cfg'. For details on the
                    values of these keys see
                    `mmdet/datasets/pipelines/formatting.py:Collect`.
                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss.
                imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                        the dimension means
                        [img_width,img_height, img_width, img_height].
                gt_masks (None | Tensor) : true segmentation masks for each box
                    used if the architecture supports a segmentation task.

            Returns:
                dict[str, Tensor]: a dictionary of loss components of all stage.
            """

            num_imgs = len(img_metas)
            num_proposals = proposal_boxes.size(1)
            imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
            all_stage_bbox_results = []
            proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
            object_feats = proposal_features
            all_stage_loss = {}
            actor_proposals = None
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)
                all_stage_bbox_results.append(bbox_results)
                if gt_bboxes_ignore is None:
                    # TODO support ignore
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                cls_pred_list = bbox_results['detach_cls_score_list']
                proposal_list = bbox_results['detach_proposal_list']
                for i in range(num_imgs):
                    normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                              imgs_whwh[i])
                    assign_result = self.bbox_assigner[stage].assign(
                        normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                        gt_labels[i], img_metas[i])
                    sampling_result = self.bbox_sampler[stage].sample(
                        assign_result, proposal_list[i], gt_bboxes[i])
                    sampling_results.append(sampling_result)
                bbox_targets = self.bbox_head[stage].get_targets(
                    sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                    True)
                cls_score = bbox_results['cls_score']
                decode_bbox_pred = bbox_results['decode_bbox_pred']

                pos_bboxes = decode_bbox_pred[bbox_targets[-1].sum(dim=-1)>0]
                pos_bboxes_inds = rois[bbox_targets[-1].sum(dim=-1)>0][:, :1]
                pos_bboxes = torch.cat((pos_bboxes_inds, pos_bboxes), dim=1)
                if actor_proposals is None:
                    actor_proposals = pos_bboxes
                else:
                    actor_proposals = torch.cat((actor_proposals, pos_bboxes), dim=0)

                single_stage_loss = self.bbox_head[stage].loss(
                    cls_score.view(-1, cls_score.size(-1)),
                    decode_bbox_pred.view(-1, 4),
                    *bbox_targets,
                    imgs_whwh=imgs_whwh)

                if self.with_mask:
                    mask_results = self._mask_forward_train(
                        stage, x, bbox_results['attn_feats'], sampling_results,
                        gt_masks, self.train_cfg[stage])
                    single_stage_loss['loss_mask'] = mask_results['loss_mask']

                for key, value in single_stage_loss.items():
                    all_stage_loss[f'stage{stage}_{key}'] = value * \
                                        self.stage_loss_weights[stage]
                object_feats = bbox_results['object_feats']

            return all_stage_loss, actor_proposals

        def simple_test(self,
                        x,
                        proposal_boxes,
                        proposal_features,
                        img_metas,
                        imgs_whwh,
                        rescale=False):

            assert self.with_bbox, 'Bbox head must be implemented.'
            assert not self.with_mask, 'Mask for action is not implemented'

            # Decode initial proposals
            num_imgs = len(img_metas)
            proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
            ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            object_feats = proposal_features
            if all([proposal.shape[0] == 0 for proposal in proposal_list]):
                # There is no proposal in the whole batch
                bbox_results = [[
                    np.zeros((0, 5), dtype=np.float32)
                    for i in range(self.bbox_head[-1].num_classes)
                ]] * num_imgs
                return bbox_results

            # Why they only get final stage's result?
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)
                object_feats = bbox_results['object_feats']
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list']

            num_classes = self.bbox_head[-1].num_classes
            det_bboxes = []
            det_labels = []

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = cls_score.softmax(-1)[..., :-1]

            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                bbox_pred_per_img = proposal_list[img_id]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, cls_score_per_img], dim=1))

            bbox_results = [
                det_bboxes[i][det_bboxes[i][:, 4] > 0.5][:, :4]
                for i in range(num_imgs)
            ]

            return bbox_results

else:
    # Just define an empty class, so that __init__ can import it.
    class SparseRoIHeadWOO:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `HEADS` from `mmdet.models`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`SparseROIHeadWOO`. ')
