import torch
import torch.nn as nn

from ..builder import NECKS

try:
    from mmdet.models import NECKS as MMDET_NECKS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@NECKS.register_module()
class WOONeck(nn.Module):
    def __init__(self,
                 feat_indices=None,
                 neck_2d=None,
                 neck_3d=None,
                 clip_len=32):
        super().__init__()
        if neck_2d is not None:
            self.neck_2d = MMDET_NECKS.build(neck_2d)
        else:
            self.neck_2d = None
        if neck_3d is not None:
            self.neck_3d = NECKS.build(neck_3d)
        else:
            self.neck_3d = None

        self.feat_indices = feat_indices
        self.clip_len = clip_len

    def forward(self, x):
        if self.neck_3d is not None:
            x = self.neck_3d(x)
        if self.feat_indices is not None:
            x = [x[i] for i in self.feat_indices]
        features_3d = x[-1]

        # Extract key frame features
        key_frame_features = []
        for feat in x:
            key_frame_features.append(feat[:, :, int(self.clip_len/2), : :])
        # FPN
        features_2d = self.neck_2d(key_frame_features)

        return features_3d, features_2d


if mmdet_imported:
    MMDET_NECKS.register_module()(WOONeck)
