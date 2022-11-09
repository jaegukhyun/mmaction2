import torch
import torch.nn.functional as F

try:
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.dense_heads import EmbeddingRPNHead
    from mmdet.core import bbox_cxcywh_to_xyxy
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class EmbeddingRPNHeadWOO(EmbeddingRPNHead):
        def _decode_init_proposals(self, imgs, img_metas):
            """Decode init_proposal_bboxes according to the size of images and
            expand dimension of init_proposal_features to batch_size.

            Args:
                imgs (list[Tensor]): List of FPN features.
                img_metas (list[dict]): List of meta-information of
                    images. Need the img_shape to decode the init_proposals.

            Returns:
                Tuple(Tensor):

                    - proposals (Tensor): Decoded proposal bboxes,
                      has shape (batch_size, num_proposals, 4).
                    - init_proposal_features (Tensor): Expanded proposal
                      features, has shape
                      (batch_size, num_proposals, proposal_feature_channel).
                    - imgs_whwh (Tensor): Tensor with shape
                      (batch_size, 4), the dimension means
                      [img_width, img_height, img_width, img_height].
            """
            proposals = self.init_proposal_bboxes.weight.clone()
            proposals = bbox_cxcywh_to_xyxy(proposals)
            num_imgs = len(imgs[0])
            imgs_whwh = []
            for meta in img_metas:
                h, w  = meta['img_shape']
                imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
            imgs_whwh = torch.cat(imgs_whwh, dim=0)
            imgs_whwh = imgs_whwh[:, None, :]

            # imgs_whwh has shape (batch_size, 1, 4)
            # The shape of proposals change from (num_proposals, 4)
            # to (batch_size ,num_proposals, 4)
            proposals = proposals * imgs_whwh

            init_proposal_features = self.init_proposal_features.weight.clone()
            init_proposal_features = init_proposal_features[None].expand(
                num_imgs, *init_proposal_features.size())
            return proposals, init_proposal_features, imgs_whwh

else:
    # Just define an empty class, so that __init__ can import it.
    class EmbeddingRPNHeadWOO:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `bbox2roi` from `mmdet.core.bbox`, '
                'or failed to import `HEADS` from `mmdet.models`, '
                'or failed to import `StandardRoIHead` from '
                '`mmdet.models.roi_heads`. You will be unable to use '
                '`EmbeddingRPNHeadWOO`. ')
