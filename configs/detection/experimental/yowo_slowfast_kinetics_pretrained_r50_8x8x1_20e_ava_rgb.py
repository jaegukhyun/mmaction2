# model setting
model = dict(
    type='YOLOXAVA',
    backbone=dict(
        type='YOWOBackbone',
        backbone_3d=dict(
            type='ResNet3dSlowFast',
            pretrained=None,
            resample_rate=4,
            speed_ratio=4,
            channel_ratio=8,
            slow_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=True,
                fusion_kernel=7,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1)),
            fast_pathway=dict(
                type='resnet3d',
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1)),
        backbone_2d=dict(
            type='CSPDarknet',
            deepen_factor=0.33,
            widen_factor=0.375),
        num_frames=32),
    neck=dict(
        type='CFAM',
        channels_2d=96,
        channels_3d=2304,
        out_channels=96,
        neck_2d=dict(
            type='YOLOXPAFPN',
            in_channels=[96, 192, 384],
            out_channels=96,
            num_csp_blocks=1),
        neck_3d=dict(
            type='SLOWFASTFUSE',
            with_temporal_pool=True)),
    bbox_head=dict(
        type='YOLOXHeadAVA',
        num_classes=81,
        in_channels=96,
        feat_channels=96,
        strides=[32]),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssignerAVA',
            center_radius=2.5)),
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(
            type='nms',
            iou_threshold=0.65)))

dataset_type = 'AVADataset'
data_root = '/home/jaeguk/workspace/data/ava/frames'
anno_root = '/home/jaeguk/workspace/data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2_sampled_12.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2_sampled_3.csv'

exclude_file_train = None
exclude_file_val = None

label_file = f'{anno_root}/ava_action_list_v2.2_sampled.pbtxt'

proposal_file_train = None
proposal_file_val = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=['entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['img_shape', 'scale_factor'],
        nested=True)
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.5,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.5,
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.00001)

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.1)
total_epochs = 20
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('./work_dirs/ava/'
            'yowo_slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb')
load_from = ('/home/jaeguk/.cache/torch/hub/checkpoints/slowfast_ava_yolox_coco.pth')
resume_from = None
find_unused_parameters = False
