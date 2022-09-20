import os
import json
import pickle
from copy import deepcopy

import cv2 as cv
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector


def get_action_label(pbtxt_file):
    action_label = {}
    with open(pbtxt_file) as pf:
        lines = pf.readlines()
        buf = []
        for line in lines:
            if 'name:' in line:
                action = line.split('\n')[0].split('name: ')[1]
                action = action.split('"')[1]
                assert len(buf) == 0
                buf.append(action)
            if 'id:' in line:
                action_id = line.split('\n')[0].split('id: ')[1]
                assert len(buf) == 1
                buf.append(action_id)
                action_label[buf[0]] = buf[1]
                buf = []

    return action_label

def main():
    mmdir = '/home/jaeguk/workspace/mmaction2'
    logdir = '/home/jaeguk/workspace/logs/action_detection'
    data_root = '/home/jaeguk/workspace/data/'
    dataset = 'ucf101-sampled'
    frame_dir = os.path.join(data_root, dataset, 'frames')
    pbtxt = os.path.join(data_root, dataset, 'annotations',
                        f'{dataset}_actionlist.pbtxt')
    json_file = os.path.join(data_root, dataset, 'annotations',
                             'instances_valid.json')
    pkl = f'{dataset}_dense_proposals_instances_valid.pkl'
    save_root = os.path.join(data_root, dataset, 'demo')
    os.makedirs(save_root, exist_ok=True)

    label_map = get_action_label(pbtxt)

    _config = os.path.join(mmdir, 'configs', 'detection', 'experimental',
        f'yowo_slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_{dataset}.py')
    config = mmcv.Config.fromfile(_config)
    val_pipeline = config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval

    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    checkpoint = os.path.join(logdir,
                              'yowo_slowfast_kinetics_pretrianed_r50_8x8x1_20e_ava_rgb_ucf101-sampled',
                              'ucf_train', 'epoch_3.pth')

    load_checkpoint(model, checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    need_proposal = False

    img_norm_cfg = config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    with open(os.path.join(data_root, f'{dataset}', 'annotations', pkl), 'rb') as fb:
        human_detections = pickle.load(fb, encoding='latin1')

    if isinstance(config.data.val.timestamp_start, int):
        _timestamp_start = config.data.val.timestamp_start
    else:
        with open(config.data.val.timestamp_start) as f:
            _timestamp_start = json.load(f)

    if isinstance(config.data.val.timestamp_end, int):
        _timestamp_end = config.data.val.timestamp_end
    else:
        with open(config.data.val.timestamp_end) as f:
            _timestamp_end = json.load(f)

    with open(json_file) as jf:
        info = json.load(jf)
        images = info['images']
        annotations = info['annotations']
        h_ratio = w_ratio = w = h = None
        for idx, image in enumerate(images):
            action, vid, frame = image['file_name'].split('/')
            image_id = image['id']
            timestamp = int(frame.split('.')[0])
            if isinstance(_timestamp_start, int):
                timestamp_start = _timestamp_start
            else:
                timestamp_start = _timestamp_start[f'{action}/{vid}']
            if isinstance(_timestamp_end, int):
                timestamp_end = _timestamp_end
            else:
                timestamp_end = _timestamp_end[f'{action}/{vid}']

            start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
            frame_inds = start_frame + np.arange(0, window_size, frame_interval)
            frame_inds = np.clip(frame_inds,
                                 timestamp_start,
                                 timestamp_end)
            imgs = [
                cv.imread(os.path.join(frame_dir, action, vid, f'{ind:05}.jpg'))
                for ind in frame_inds
            ]
            if w == None:
                h, w, _  = imgs[0].shape
            #new_w, new_h = mmcv.rescale_size((w, h), (256, 256))
            new_w, new_h = (256, 256)
            if h_ratio == None:
                h_ratio, w_ratio = new_h / h, new_w / w
                scale_factor = np.array([w_ratio, h_ratio], dtype=np.float32)
            imgs = [mmcv.imresize(img, (new_w, new_h)).astype(np.float32)
                    for img in imgs]
            _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
            # THWC -> CTHW -> 1CTHW
            input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).cuda()

            if need_proposal:
                if f'{action}/{vid},{timestamp:05}' not in human_detections.keys():
                    proposal = np.array([[0, 0, h, w, 1]], dtype=np.float32)
                else:
                    proposal = human_detections[f'{action}/{vid},{timestamp:05}']
                    multiplier = np.asarray([[w, h, w, h]], dtype=np.float32)
                    proposal[:, :4] = proposal[:, :4] * multiplier
                cp_proposal = deepcopy(proposal)
                proposal[:, 0:4:2] *= w_ratio
                proposal[:, 1:4:2] *= h_ratio
                proposal = torch.from_numpy(proposal[:, :4]).cuda()

                with torch.no_grad():
                    result = model(
                        return_loss=False,
                        img=[input_tensor],
                        img_metas=[[dict(img_shape=(new_h, new_w))]],
                        proposals=[[proposal]]
                    )
                    result = result[0] # only handle single action
                    score = label = -1
                    for i in range(len(result)):
                        try:
                            _score = result[i][0][4]
                        except:
                            continue
                        if _score > score:
                            label = i + 1
                            score = _score
                    for k, v in label_map.items():
                        if int(v) == label:
                            _label = k
                    print(f'[answer|perdict, score]: {action}|{_label}, {score:.2f}')

                    cv_img = cv.imread(os.path.join(frame_dir, image['file_name']))
                    x1,y1,x2,y2,p = cp_proposal[0]
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv.rectangle(cv_img, p1, p2, (0, 255, 0), 1)
                    cv.putText(cv_img, f'{_label}: {score:.2f}', (p1[0] + 15, p1[1] + 15),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    cv.imwrite(os.path.join(save_root, f'{action}_{idx}.jpg'), cv_img)

            else:
                cv_img = cv.imread(os.path.join(frame_dir, image['file_name']))
                _label = -1
                score = 0
                with torch.no_grad():
                    result = model(
                        return_loss=False,
                        img=[input_tensor],
                        img_metas=[[dict(img_shape=(new_h, new_w),
                                         scale_factor=scale_factor)]],
                    )
                    result = result[0] # only handle single action
                    for i in range(len(result)):
                        if len(result[i]) == 0:
                            continue
                        x1,y1,x2,y2,score = result[i][0]
                        for k, v in label_map.items():
                            if int(v) == i + 1:
                                _label = k

                        p1 = (int(x1 * new_w / w_ratio), int(y1 * new_h / h_ratio))
                        p2 = (int(x2 * new_w / w_ratio), int(y2 * new_h / h_ratio))
                        if score > 0.3:
                            cv.rectangle(cv_img, p1, p2, (0, 255, 0), 1)
                            cv.putText(cv_img, f'{_label}: {score:.2f}', (p1[0] + 15, p1[1] + 15),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                print(f'[answer|perdict, score]: {action}|{_label}, {score:.2f}')
                #for annot in annotations:
                #    if annot['image_id'] == image_id:
                #        x1, y1, w, h = annot['bbox']
                #        cv.rectangle(cv_img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 1)
                cv.imwrite(os.path.join(save_root, f'{action}_{idx}.jpg'), cv_img)


if __name__ == '__main__':
    main()
