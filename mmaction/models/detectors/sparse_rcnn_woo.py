import torch
import numpy as np

try:
    from mmdet.models.builder import DETECTORS
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.detectors import SparseRCNN
    mmdet_imported = True
except(ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:
    @DETECTORS.register_module()
    class SparseRCNNWOO(SparseRCNN):
        def __init__(self,
                     roi_head2d=None,
                     roi_head3d=None,
                     *args,
                     **kwargs):
            train_cfg = kwargs.get('train_cfg', None)
            test_cfg = kwargs.get('test_cfg', None)
            pretrained = kwargs.get('pretrained', None)

            roi_head2d_train_cfg = train_cfg.roi_head2d if train_cfg is not None else None
            roi_head2d.update(train_cfg=roi_head2d_train_cfg)
            roi_head2d.update(test_cfg=test_cfg.rcnn)
            roi_head2d.pretrained = pretrained

            roi_head3d_train_cfg = train_cfg.roi_head3d if train_cfg is not None else None
            roi_head3d.update(train_cfg=roi_head3d_train_cfg)
            roi_head3d.update(test_cfg=test_cfg.rcnn)
            roi_head3d.pretrained = pretrained

            super(SparseRCNNWOO, self).__init__(*args, **kwargs)

            self.roi_head2d = MMDET_HEADS.build(roi_head2d)
            self.roi_head3d = MMDET_HEADS.build(roi_head3d)

        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          gt_masks=None,
                          proposals=None,
                          **kwargs):
            """Forward function of SparseR-CNN and QueryInst in train stage.

            Args:
                img (Tensor): of shape (N, C, H, W) encoding input images.
                    Typically these should be mean centered and std scaled.
                img_metas (list[dict]): list of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    :class:`mmdet.datasets.pipelines.Collect`.
                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor): specify which bounding
                    boxes can be ignored when computing the loss.
                gt_masks (List[Tensor], optional) : Segmentation masks for
                    each box. This is required to train QueryInst.
                proposals (List[Tensor], optional): override rpn proposals with
                    custom proposals. Use when `with_rpn` is False.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """

            assert proposals is None, 'Sparse R-CNN and QueryInst ' \
                'do not support external proposals'

            gt_labels_2d = [torch.zeros(gt_label.shape[0], dtype=torch.long).to(gt_label.device) for gt_label in gt_labels]
            features_3d, features_2d = self.extract_feat(img)
            proposal_boxes, proposal_features, imgs_whwh = self.rpn_head.forward_train(features_2d, img_metas)

            roi_losses, _actor_proposals = self.roi_head2d.forward_train(
                features_2d,
                proposal_boxes,
                proposal_features,
                img_metas,
                gt_bboxes,
                gt_labels_2d,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                imgs_whwh=imgs_whwh)

            batch_size = img.shape[0]
            actor_proposals = []
            for idx in range(batch_size):
                actor_proposals.append(_actor_proposals[_actor_proposals[:, 0]==idx][:, 1:])

            action_cls_loss = self.roi_head3d.forward_train(
                features_3d,
                img_metas,
                actor_proposals,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                gt_masks=None,
                **kwargs
            )
            roi_losses.update(action_cls_loss)

            return roi_losses

        def simple_test(self, img, img_metas, rescale=False):
            device = img.device
            features_3d, features_2d = self.extract_feat(img)
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.simple_test_rpn(features_2d, img_metas)

            for img_meta in img_metas:
                img_meta['ori_shape'] = img_meta.pop('original_shape')

            results = self.roi_head2d.simple_test(
                features_2d,
                proposal_boxes,
                proposal_features,
                img_metas,
                imgs_whwh=imgs_whwh,
                rescale=rescale)

            if results[0].shape[0] == 0:
                bbox_results = [[
                    np.zeros((0, 5), dtype=np.float32)
                    for i in range(self.roi_head2d.bbox_head[-1].num_classes)
                ]] * len(results)
                return bbox_results

            results = self.roi_head3d.simple_test(features_3d,
                                                  results,
                                                  img_metas,
                                                  rescale=rescale)

            return results


else:
    # Just define an empty class, so that __init__ can import it.
    class SparseRCNNWOO:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `HEADS` from `mmdet.models`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`SparseRCNNWOO`. ')
