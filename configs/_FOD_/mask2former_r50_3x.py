_base_ = [
    '../_base_/models/mask2former_r50.py',
    '../_base_/datasets/FOD.py',
    '../_base_/schedules/schedule_3x.py', 
    '../_base_/runtime.py'
]

# remove batch_augments
data_preprocessor = dict(
    _delete_=True,
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    )

num_things_classes = 3
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=200,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]), 
        ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
)

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_ = True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

work_dir = './work_dirs/FOD/{{fileBasenameNoExtension}}'