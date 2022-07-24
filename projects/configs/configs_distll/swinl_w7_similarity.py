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
        checkpoint='nfs/mobileAI2022_final/distill/teacher_swint_full_600e/epoch_400.pth'),
    gt_target_shape=(480, 640),
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
        type='DenseDepthHeadLightMobile',
        in_index=(0, 1, 2, 3),
        debug=True,
        with_loss_depth_grad=True,
        loss_depth_grad=dict(
            type='GradDepthLoss', valid_mask=True, loss_weight=0.2),
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
        in_channels=[192, 384, 768, 1536],
        up_sample_channels=[0, 48, 192, 768],
        logits_dim=24, # last one
        # align_corners=False, # for upsample
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

student_model_cfg = dict(
    type='DepthEncoderDecoderMobile',
    gt_target_shape=(480, 640),
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='nfs/mobileAI2022_final/final_models/model1_650e/epoch_650.pth'),
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

distill_weight=10
model=dict(
    type='DistillWrapper',
    upsample_type='bilinear',
    distill=True,
    ema=False,
    teacher_depther_cfg=teacher_model_cfg,
    student_depther_cfg=student_model_cfg,
    teacher_select_de_index=(0, 2, 3),
    student_select_de_index=(0, 2, 3),
    layer_weights=(1, 1, 1),
    distill_loss=dict(type='SimilarityMSELoss', loss_weight=distill_weight),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# dataset settings Only for test
dataset_type = 'MobileAI2022Dataset'
data_root = 'data/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCropV2', pick_mode=True, crop_size=[(240, 384), (360, 512), (480, 640)]),
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
    dict(type='ColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='NormalizeDistill', 
         teacher_norm_mean=[123.675, 116.28, 103.53], 
         teacher_norm_std=[58.395, 57.12, 57.375], 
         teacher_norm_to_rgb=True, 
         student_norm_mean=[127.5, 127.5, 127.5], 
         student_norm_std=[127.5, 127.5, 127.5], 
         student_norm_to_rgb=True,
         norm_mean=[127.5, 127.5, 127.5], 
         norm_std=[127.5, 127.5, 127.5], 
         norm_to_rgb=True),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_teacher', 'img_student', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeImg', img_scale_ori=(480, 640), img_scale_target=(128, 160)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(0, 0),
        transforms=[
            dict(type='NormalizeDistill', 
                 teacher_norm_mean=[123.675, 116.28, 103.53], 
                 teacher_norm_std=[58.395, 57.12, 57.375], 
                 teacher_norm_to_rgb=True, 
                 student_norm_mean=[127.5, 127.5, 127.5], 
                 student_norm_std=[127.5, 127.5, 127.5], 
                 student_norm_to_rgb=True,
                 norm_mean=[127.5, 127.5, 127.5], 
                 norm_std=[127.5, 127.5, 127.5], 
                 norm_to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'img_teacher', 'img_student']),
            dict(type='Collect', keys=['img', 'img_teacher', 'img_student']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    # train=dict(
    #     type=dataset_type,
    #     pipeline=train_pipeline,
    #     data_root='data/train',
    #     test_mode=False,
    #     min_depth=1e-3,
    #     depth_scale=1000), # convert to meters
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root='data/trainval',
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
max_lr=4e-3
optimizer = dict(type='Adam', lr=max_lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)
lr_config = dict(policy='poly', power=0.9, min_lr=max_lr*1/4, by_epoch=False, warmup='linear', warmup_iters=1500, warmup_ratio=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=4, interval=100)
evaluation = dict(by_epoch=True, interval=25, pre_eval=True)

find_unused_parameters=True