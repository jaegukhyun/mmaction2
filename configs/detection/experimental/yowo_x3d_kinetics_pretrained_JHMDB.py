# model setting
model = dict(
    type='YOLOXAVA',
    backbone=dict(
        type='YOWOBackbone',
        backbone_3d=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
        backbone_2d=dict(
            type='CSPDarknet',
            deepen_factor=0.33,
            widen_factor=0.375),
        num_frames=32),
    neck=dict(
        type='CFAM',
        channels_2d=96,
        channels_3d=432,
        out_channels=96,
        neck_2d=dict(
            type='YOLOXPAFPN',
            in_channels=[96, 192, 384],
            out_channels=96,
            num_csp_blocks=1)),
    bbox_head=dict(
        type='YOLOXHeadAVA',
        num_classes=22,
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

dataset_type = 'JHMDBDataset'
data_root = '/home/jaeguk/workspace/data/JHMDB/frames'
anno_root = '/home/jaeguk/workspace/data/JHMDB/annotations'

ann_file_train = f'{anno_root}/JHMDB_train_105.csv'
ann_file_val = f'{anno_root}/JHMDB_valid_42.csv'

exclude_file_train = None
exclude_file_val = None

label_file = f'{anno_root}/JHMDB_actionlist.pbtxt'

proposal_file_train = None
proposal_file_val = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=1),
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
        type='SampleAVAFrames', clip_len=32, frame_interval=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['img_shape', 'scale_factor', 'entity_ids'],
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
        data_prefix=data_root,
        filename_tmpl='{:05}.png',
        timestamp_start=1,
        timestamp_end='/home/jaeguk/workspace/data/JHMDB/annotations/JHMDB_timestamp.json',
        start_index=1,
        num_classes=22,
        fps=1
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.5,
        data_prefix=data_root,
        filename_tmpl='{:05}.png',
        timestamp_start=1,
        timestamp_end='/home/jaeguk/workspace/data/JHMDB/annotations/JHMDB_timestamp.json',
        start_index=1,
        num_classes=22,
        fps=1
    )
)
data['test'] = data['val']

optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.00001)

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[40],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.1)
total_epochs = 50
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
load_from = ('/home/jaeguk/.cache/torch/hub/checkpoints/x3d_kinetics_yolox_coco.pth')
resume_from = None
find_unused_parameters = False

