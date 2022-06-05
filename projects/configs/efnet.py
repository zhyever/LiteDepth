_base_ = [
    '../../Monocular-Depth-Estimation-Toolbox/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/toolbox_plugin/'

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoderTF',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50')),
    decode_head=dict(
        type='DenseDepthHead',
        scale_up=True,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[64, 256, 512, 1024, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset settings Only for test
dataset_type = 'MobileAI2022Dataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    # dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(416, 544)),
    dict(type='ColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(480, 640)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(0, 0),
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root='data/train',
        test_mode=False,
        min_depth=1e-3,
        depth_scale=1000), # convert to meters
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='data/local_val',
        test_mode=False,
        min_depth=1e-3,
        depth_scale=1000),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='data/online_val',
        test_mode=True,
        min_depth=1e-3,
        depth_scale=1000),
    # test=dict(
    #     type=dataset_type,
    #     pipeline=test_pipeline,
    #     data_root='data/local_val',
    #     test_mode=False,
    #     min_depth=1e-3,
    #     depth_scale=1000)
)

# optimizer
max_lr=1e-4
optimizer = dict(type='AdamW', lr=max_lr, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=6)
evaluation = dict(by_epoch=True, interval=3, pre_eval=True)
