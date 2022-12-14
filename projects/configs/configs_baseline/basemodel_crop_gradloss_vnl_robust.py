


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
    gt_target_shape=(480, 640),
    backbone=dict(
        type="mmcls.TIMMBackbone",
        pretrained=True,
        model_name="tf_mobilenetv3_small_minimal_100",
        features_only=True),
    decode_head=dict(
        type='DenseDepthHeadLightMobile',
        in_index=(1, 2, 3, 4),
        debug=True,
        with_loss_depth_grad=True,
        loss_depth_grad=dict(
            type='GradDepthLoss', valid_mask=True, loss_weight=0.25),
        with_loss_vnl=True,
        loss_vnl=dict(
            type='VNLLoss', 
            focal_x=5.1885790117450188e+02, 
            focal_y=5.1946961112127485e+02, 
            input_size=(480, 640),
            delta_cos=0.867, 
            delta_diff_x=0.01,
            delta_diff_y=0.01,
            delta_diff_z=0.01,
            delta_z=0.1, #mask invalid depth
            sample_ratio=0.15,
            loss_weight=2.5),
        with_loss_robust=True,
        loss_robust=dict(
            type='RobustLoss', 
            loss_weight=0.6,
            log=False),
        scale_up=False,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[16, 24, 48, 96],
        up_sample_channels=[0, 8, 24, 72],
        logits_dim=24,
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCropV2', pick_mode=True, crop_size=[(240, 384), (360, 512), (480, 640)]),
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
    dict(type='ColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
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
    # train=dict(
    #     type=dataset_type,
    #     pipeline=train_pipeline,
    #     data_root='data/trainval',
    #     test_mode=False,
    #     min_depth=1e-3,
    #     depth_scale=1000), # convert to meters
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='data/local_val',
        test_mode=False,
        min_depth=1e-3,
        depth_scale=1000),
    # test=dict(
    #     type=dataset_type,
    #     pipeline=test_pipeline,
    #     data_root='data/online_val',
    #     test_mode=True,
    #     min_depth=1e-3,
    #     depth_scale=1000),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='data/local_val',
        test_mode=False,
        min_depth=1e-3,
        depth_scale=1000)
)

# optimizer
max_lr=4e-3
optimizer = dict(type='Adam', lr=max_lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)
lr_config = dict(policy='poly', power=0.9, min_lr=max_lr*1/4, by_epoch=False, warmup='linear', warmup_iters=1500, warmup_ratio=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=600)
runner = dict(type='EpochBasedRunner', max_epochs=650)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=50)
# evaluation = dict(by_epoch=True, interval=999, pre_eval=True)
evaluation = dict(by_epoch=True, interval=50, pre_eval=True)

find_unused_parameters=True