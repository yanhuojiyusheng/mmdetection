_base_ =    [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/FOD.py',
    '../_base_/schedules/schedule_3x.py', 
    '../_base_/runtime.py'
]

# modify num classes
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        bbox_head=dict(num_classes=3), 
        mask_head=dict(
            num_classes=3),
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=3)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)))

work_dir = './work_dirs/FOD/{{fileBasenameNoExtension}}'