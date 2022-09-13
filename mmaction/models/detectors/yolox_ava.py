import torch

try:
    from mmdet.models.builder import DETECTORS
    from mmdet.models.detectors import YOLOX, SingleStageDetector
    mmdet_imported = True
except(ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:
    @DETECTORS.register_module()
    class YOLOXAVA(YOLOX):
        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None):
            """
            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                    Typically these should be mean centered and std scaled.
                img_metas (list[dict]): A List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    :class:`mmdet.datasets.pipelines.Collect`.
                gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                    image in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): Class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                    boxes can be ignored when computing the loss.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            losses = super(YOLOX, self).forward_train(img,
                                                      img_metas,
                                                      gt_bboxes,
                                                      gt_labels,
                                                      gt_bboxes_ignore)

            # random resizing
            if (self._progress_in_iter + 1) % self._random_size_interval == 0:
                self._input_size = self._random_resize(device=img.device)
            self._progress_in_iter += 1

            return losses

else:
    # Just define an empty class, so that __init__ can import it.
    class YOLOXAVA:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `HEADS` from `mmdet.models`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`YOLOXAVA`. ')
