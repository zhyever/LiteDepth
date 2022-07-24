# This config file is used to convert model from pytorch to tflite only.

plugin=True
plugin_dir='projects/toolbox_plugin/'

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoderMobileMerge',
    downsample_target=(128, 160),
    # downsample_target=(96, 128),
    backbone=dict(
        type="mmcls.TIMMBackbone",
        pretrained=True,
        model_name="tf_mobilenetv3_small_minimal_100",
        features_only=True),
    decode_head=dict(
        type='DenseDepthHeadLightMobile',
        in_index=(1, 2, 3, 4),
        debug=False,
        with_loss_depth_grad=True,
        loss_depth_grad=dict(
            type='GradDepthLoss', valid_mask=True, loss_weight=0.3),
        scale_up=False,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[16, 24, 48, 96],
        up_sample_channels=[0, 8, 24, 72],
        logits_dim=24,
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
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True) # only to rgb

train_pipeline = []
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
        data_root='data/local_val',
        test_mode=False,
        min_depth=1e-3,
        depth_scale=1000)
)
