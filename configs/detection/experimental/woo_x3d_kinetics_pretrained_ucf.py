num_stages = 6
num_proposals = 100
model = dict(
    type='SparseRCNNWOO',
    pretrained=False,
    backbone=dict(
        type='X3D',
        return_intermediate=True,
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2),
    neck=dict(
        type='WOONeck',
        neck_2d=dict(
            type='FPN',
            in_channels=[48, 96, 192, 432],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=4),
        neck_3d=None,
        feat_indices=[1,2,3,4]),
    rpn_head=dict(
        type='EmbeddingRPNHeadWOO',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head2d=dict(
        type='SparseRoIHeadWOO',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                loss_iou=dict(type='GIoULoss', loss_weight=1.0),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)]),
    roi_head3d = dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=432,
            num_classes=11,
            multilabel=False,
            dropout_ratio=0.5)),
    train_cfg=dict(
        rpn=None,
        roi_head2d=[
            dict(
                assigner=dict(
                    type='HungarianAssignerWOO',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)],
        roi_head3d=dict(
            # Do we really need assigner in roi_head3d?
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
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals, action_thr=0.002)))

dataset_type = 'JHMDBDataset'
data_root = '/home/jaeguk/workspace/data/ucf101-sampled/frames'
anno_root = '/home/jaeguk/workspace/data/ucf101-sampled/annotations'

ann_file_train = f'{anno_root}/ucf101-sampled_train_50.csv'
ann_file_val = f'{anno_root}/ucf101-sampled_valid_20.csv'

exclude_file_train = None
exclude_file_val = None

label_file = f'{anno_root}/ucf101-sampled_actionlist.pbtxt'

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
        meta_keys=['entity_ids', 'img_shape'])
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
        meta_keys=['original_shape','img_shape', 'scale_factor', 'gt_bboxes', 'gt_labels'],
        nested=True)
]

data = dict(
    videos_per_gpu=4,
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
        filename_tmpl='{:05}.jpg',
        timestamp_start=1,
        timestamp_end='/home/jaeguk/workspace/data/ucf101-sampled/annotations/ucf101-sampled_timestamp.json',
        start_index=1,
        num_classes=11,
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
        filename_tmpl='{:05}.jpg',
        timestamp_start=1,
        timestamp_end='/home/jaeguk/workspace/data/ucf101-sampled/annotations/ucf101-sampled_timestamp.json',
        start_index=1,
        num_classes=11,
        fps=1
    )
)
data['test'] = data['val']


optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1,
    warmup_ratio=0.1)
total_epochs = 20
checkpoint_config = dict(save_last=True, max_keep_ckpts=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('./work_dirs/ava/'
            'yowo_slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb')
load_from = ('/home/jaeguk/.cache/torch/hub/checkpoints/'
             'woo_x3d_kinetics_sparsercnn_coco.pth')
resume_from = None
find_unused_parameters = False
