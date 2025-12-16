default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    checkpoint=dict(type='CheckpointHook',by_epoch=True, interval=1, max_keep_ckpts=1,save_best='coco/segm_mAP',rule = 'greater'),
    checkloss = dict(type='CheckInvalidLossHook',interval=1),
    visualization=dict(
        draw_bbox=False,draw_gt=False,draw=True, test_out_dir='testresult',interval=20000, type='DetVisualizationHook')
    # You can adjust the interval parameter to increase the number of validation images saved during the training process.
    )

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
