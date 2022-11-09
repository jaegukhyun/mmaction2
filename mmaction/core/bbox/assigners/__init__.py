# Copyright (c) OpenMMLab. All rights reserved.
from .max_iou_assigner_ava import MaxIoUAssignerAVA
from .sim_ota_assigner_ava import SimOTAAssignerAVA
from .hungarian_assigner_woo import HungarianAssignerWOO

__all__ = ['MaxIoUAssignerAVA', 'SimOTAAssignerAVA', 'HungarianAssignerWOO']
