# model setting
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='MViT',
        spatial_size=256,
        num_frames=32,
        enable_detection=True,
        zero_decay_pos_cls=False,
        use_abs_pos=False,
        rel_pos_spatial=True,
        rel_pos_temporal=True,
        depth=16,
        num_heads=1,
        embed_dim=96,
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.2,
        norm="layernorm",
        mode="conv",
        cls_embed_on=True,
        _dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        _head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        pool_kvq_kernel=[3, 3, 3],
        pool_kv_stride_adaptive=[1, 8, 8],
        pool_q_stride=[[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2],
                      [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1],
                      [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
                      [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]],
        dropout_rate=0.0,
        dim_mul_in_att=True,
        residual_pooling=True,
    ),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=768,
            num_classes=81,
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

dataset_type = 'AVADataset'
data_root = '/home/jaeguk/workspace/data/ava/frames'
anno_root = '/home/jaeguk/workspace/data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'

label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

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
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
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
    videos_per_gpu=2,
    workers_per_gpu=16,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1)
total_epochs = 10
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('/home/jaeguk/workspace/logs/action_detection/'
            'mvit_kinetics_pretrained_ava_rgb/ava/')
load_from = (
    '/home/jaeguk/.cache/torch/hub/checkpoints/'
    'MViTv2_S_16x4_k400_f302660347_mmaction2.pyth'
)
resume_from = None
find_unused_parameters = False

