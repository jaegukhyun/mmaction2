import torch
import sys


def main(backbone_weight, object_detector_weight, dst, style):
    try:
        backbone_state_dict = torch.load(backbone_weight)['state_dict']
    except:
        backbone_state_dict = torch.load(backbone_weight)
    object_detector_state_dict = torch.load(object_detector_weight)['state_dict']
    new_state_dict = {}
    if style == 'yowo':
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
    elif style == 'woo':
        for k, v in backbone_state_dict.items():
            if 'backbone' in k:
                new_state_dict[k] = v
            elif 'roi_head' in k:
                k = k.replace('roi_head', 'roi_head3d')
                new_state_dict[k] = v

        for k, v in object_detector_state_dict.items():
            if 'neck' in k:
                k = k.replace('neck', 'neck.neck_2d')
                new_state_dict[k] = v
            elif 'rpn_head' in k:
                new_state_dict[k] = v
            elif 'roi_head' in k:
                k = k.replace('roi_head', 'roi_head2d')
                new_state_dict[k] = v
    else:
        raise NotImplementedError('Supported style is YOWO, and WOO')

    torch.save(new_state_dict, dst)


if __name__ == '__main__':
    print('Usage: python merge_weights.py backbone_weight object_detector_weight destination style')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print('Done!')

