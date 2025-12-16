# training schedule for 3x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1,val_begin=0)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        gamma=0.1,
        milestones=[
            24, 
            30,
        ],)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
        optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
