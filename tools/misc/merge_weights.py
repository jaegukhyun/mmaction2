import torch
import sys


def main(backbone_weight, object_detector_weight, dst):
    backbone_state_dict = torch.load(backbone_weight)['state_dict']
    object_detector_state_dict = torch.load(object_detector_weight)['state_dict']
    new_state_dict = {}

    for k, v in backbone_state_dict.items():
        if 'backbone' in k:
            k = k.replace('backbone', 'backbone.backbone_3d')
            new_state_dict[k] = v

    for k, v in object_detector_state_dict.items():
        if 'backbone' in k:
            k = k.replace('backbone', 'backbone.backbone_2d')
            new_state_dict[k] = v
        elif 'neck' in k:
            k = k.replace('neck', 'neck.neck_2d')
            new_state_dict[k] = v
        elif 'bbox_head' in k:
            new_state_dict[k] = v

    torch.save(new_state_dict, dst)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

