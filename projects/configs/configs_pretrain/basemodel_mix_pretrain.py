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
        in_index=(0, 1, 2, 3, 4),
        debug=False,
        with_loss_depth_grad=True,
        loss_depth_grad=dict(
            type='GradDepthLoss', valid_mask=True, loss_weight=0.3),
        # scale_up=True,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[16, 16, 24, 48, 96],
        up_sample_channels=[0, 16, 24, 48, 96],
        channels=32, # last one
        # align_corners=False, # for upsample
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, warm_up=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'NYUDataset'
data_root = 'data/nyu/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True) # incpt
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomCrop', crop_size=(416, 544)),
    dict(type='RandomCropV2', pick_mode=True, crop_size=[(120, 256), (240, 384), (360, 512), (480, 640)]), # 
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
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
dataset_nyu=dict(
    type=dataset_type,
    data_root=data_root,
    depth_scale=1000/4,
    split='nyu_train.txt',
    pipeline=train_pipeline,
    garg_crop=False,
    eigen_crop=True,
    min_depth=1e-3,
    max_depth=40)

dataset_type_kitti = 'KITTIDataset'
data_root_kitti = 'data/kitti/'
train_pipeline_kitti = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomCrop', crop_size=(416, 544)),
    dict(type='RandomCropV2', pick_mode=True, crop_size=[(120, 256), (240, 384), (360, 512), (480, 640)]), # 
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
dataset_kitti=dict(
    type=dataset_type_kitti,
    data_root=data_root_kitti,
    img_dir='input',
    ann_dir='gt_depth',
    depth_scale=256*2,
    split='kitti_eigen_train.txt',
    pipeline=train_pipeline_kitti,
    garg_crop=True,
    eigen_crop=False,
    min_depth=1e-3,
    max_depth=80)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train = [
        dataset_nyu,
        dataset_kitti
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000/4,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=40),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000/4,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=40))

# optimizer
max_lr=3e-3
optimizer = dict(type='Adam', lr=max_lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)

lr_config = dict(policy='poly', power=0.9, min_lr=max_lr*1e-2, by_epoch=False, warmup='linear', warmup_iters=1000)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=10)
evaluation = dict(by_epoch=True, interval=10, pre_eval=True)
