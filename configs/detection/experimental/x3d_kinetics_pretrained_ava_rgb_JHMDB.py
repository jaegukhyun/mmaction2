# model setting
model = dict(
    type='FastRCNN',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=432,
            num_classes=22,
            multilabel=True,
            dropout_ratio=0.5)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

dataset_type = 'JHMDBDataset'
data_root = '/home/jaeguk/workspace/data/JHMDB/frames'
anno_root = '/home/jaeguk/workspace/data/JHMDB/annotations'

ann_file_train = f'{anno_root}/JHMDB_train_105.csv'
ann_file_val = f'{anno_root}/JHMDB_valid_42.csv'

exclude_file_train = None
exclude_file_val = None

label_file = f'{anno_root}/JHMDB_actionlist.pbtxt'

proposal_file_train = f'{anno_root}/JHMDB_dense_proposals_instances_train.pkl'
proposal_file_val = f'{anno_root}/JHMDB_dense_proposals_instances_valid.pkl'

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
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
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

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00001)

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.1)
total_epochs = 20
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('/home/jaeguk/workspace/logs/action_detection/'
            'x3d_kinetics_pretrained_ava_rgb/cosine/')
load_from = (
    'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/'
    'x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
)
#load_from = (
#    '/home/jaeguk/.cache/torch/hub/checkpoints/x3d_m_ava_16.6.pth'
#)
resume_from = None
find_unused_parameters = False

