_base_ = [
    '../_base_/models/solov2_r50_fpn.py',
    '../_base_/datasets/FOD.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/runtime.py'
]

# modify num classes
model = dict(
    mask_head=dict(
        num_classes=3))

work_dir = './work_dirs/FOD/{{fileBasenameNoExtension}}'
