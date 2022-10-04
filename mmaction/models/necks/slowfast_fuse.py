import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS

try:
    from mmdet.models import NECKS as MMDET_NECKS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@NECKS.register_module()
class SLOWFASTFUSE(nn.Module):
    def __init__(self,
                 with_temporal_pool=True,
                 temporal_pool_mode='avg'):
        super().__init__()
        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

    def forward(self, feat):
        assert len(feat) == 2

        maxT = max([x.shape[2] for x in feat])
        max_shape = (maxT, ) + feat[0].shape[3:]
        # resize each feat to the largest shape (w. nearest)
        feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        return feat


if mmdet_imported:
    MMDET_NECKS.register_module()(SLOWFASTFUSE)
