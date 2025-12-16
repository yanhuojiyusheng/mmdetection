# dataset settings
dataset_type = 'CocoDataset'
base_root = ''
data_root = base_root+'data/FOD/'
test_root = base_root+'data/ori/'  
test_root = data_root  

metainfo = {
    'classes': ('part', 'whole', 'fragment'),
    'palette': [
        (0, 255, 255),    
        (255, 191, 0),     
        (255, 0, 255)      
    ]
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),

    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,  # 如果设置为 True，dataloader 
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
        )


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=test_root,
         metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
         metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))



val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['segm'],
    classwise=True,
    format_only=False,
    backend_args=backend_args,
    outfile_prefix='./work_dirs/val_results/',
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=test_root + 'annotations/instances_test.json',
    metric=['segm'],
    format_only=False,
    classwise=True,
    outfile_prefix = './work_dirs/test_results/',
    backend_args=backend_args,
    )

