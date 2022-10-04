import torch
import torch.nn as nn

from ..builder import BACKBONES

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@BACKBONES.register_module()
class YOWOBackbone(nn.Module):
    def __init__(self, backbone_2d, backbone_3d, num_frames):
        super().__init__()
        self.backbone_2d = MMDET_BACKBONES.build(backbone_2d)
        self.backbone_3d = BACKBONES.build(backbone_3d)
        self.key_frame = num_frames // 2


    def forward(self, x):
        # TODO Verify key frame
        feature_2d = self.backbone_2d(x[:, :, self.key_frame, :, :])
        feature_3d = self.backbone_3d(x)

        return [feature_3d, feature_2d]


if mmdet_imported:
    MMDET_BACKBONES.register_module()(YOWOBackbone)
