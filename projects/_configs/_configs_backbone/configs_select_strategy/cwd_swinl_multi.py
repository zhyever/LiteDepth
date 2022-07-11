_base_ = [
    '../../../Monocular-Depth-Estimation-Toolbox/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/toolbox_plugin/'

# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

teacher_model_cfg = dict(
    type='DepthEncoderDecoderMobile',
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='nfs/checkpoints/swinl_w7_22k_align_decoder_extendup.pth'),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        pretrain_img_size=224,
        patch_size=4,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official'), # the most small version
    decode_head=dict(
        type='DenseDepthHeadMobile',
        scale_up=True,
        min_depth=1e-3,
        max_depth=40,
        in_channels=[192, 384, 768, 1536],
        up_sample_channels=[24, 32, 64, 96],
        channels=16, # last one
        extend_up_conv_num=1,
        # align_corners=False, # for upsample
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

student_model_cfg = dict(
    type='DepthEncoderDecoderMobile',
    backbone=dict(
        type="mmcls.TIMMBackbone",
        pretrained=True,
        model_name="tf_mobilenetv3_small_minimal_100",
        features_only=True),
    decode_head=dict(
        type='DenseDepthHeadMobile',
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

model=dict(
    type='DistillWrapper',
    teacher_select_de_index=(0, 1, 2),
    student_select_de_index=(0, 1, 2),
    layer_weights=(1, 1, 1),
    teacher_depther_cfg=teacher_model_cfg,
    student_depther_cfg=student_model_cfg,
    distill_loss=dict(type='ChannelWiseDivergence', loss_weight=0.5, tau=4),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    img_norm_cfg_teacher=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    img_norm_cfg_student=dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
)

# dataset settings Only for test
dataset_type = 'MobileAI2022Dataset'
data_root = 'data/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # teacher
# img_norm_cfg = dict(
#     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True) # incpt stu
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)
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
max_lr=3e-4
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
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=10)
evaluation = dict(by_epoch=True, interval=10, pre_eval=True)

find_unused_parameters=True