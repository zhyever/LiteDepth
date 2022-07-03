_base_ = [
    '../../../Monocular-Depth-Estimation-Toolbox/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/toolbox_plugin/'

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoderMobile',
    backbone=dict(
        type="mmcls.TIMMBackbone",
        pretrained=True,
        model_name="tf_mobilenetv3_small_minimal_100",
        features_only=True),
    decode_head=dict(
        type='DenseDepthHeadLightMobile',
        debug=False,
        with_loss_depth_grad=True,
        loss_depth_grad=dict(
            type='GradDepthLoss', valid_mask=True, loss_weight=0.5),
        with_loss_ssim=True,
        loss_ssim=dict(
            type='SSIMDepthLoss', valid_mask=True, loss_weight=0.5),
        scale_up=True,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[16, 16, 24, 48, 576],
        up_sample_channels=[16, 24, 32, 64, 96],
        channels=16, # last one
        # align_corners=False, # for upsample
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
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True) # incpt
# crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCropV2', pick_mode=True, crop_size=[(384, 512), (480, 640)]),
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
max_lr=3e-4
optimizer = dict(type='AdamW', lr=max_lr, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=max_lr,
#     div_factor=25,
#     final_div_factor=100,
#     by_epoch=False,
# )
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='poly', power=0.9, min_lr=max_lr*1e-2, by_epoch=False, warmup='linear', warmup_iters=1000)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=10)
evaluation = dict(by_epoch=True, interval=10, pre_eval=True)
