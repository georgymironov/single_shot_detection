seed = 23
use_gpu = True

model = {
    'base': {
        'name': 'mobilenet_v2_10',
        'weight': './weights/mobilenet_v2_keras.pt',
        'batch_norm': {
            'eps': 1e-3,
        }
    },
    'detector': {
        'use_depthwise': True,
        'source_layers': ((13, 'expand_relu'), 18, 's', 's', 's', 's'),
        'extra_layer_depth': (None, None, 512, 256, 256, 128),
        'min_scale': 0.1,
        'max_scale': 1.05,
        'num_scales': 6,
        'aspect_ratios': [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 0.3333]] * 3 + [[1, 2, 0.5]] * 2,
        'num_branches': [2, 2, 1, 1, 1, 1]
    }
}

box_coder = {
    'xy_scale': 10.0,
    'wh_scale': 5.0,
}

loss = {
    'classification_weight': 1.0,
    'localization_weight': 1.0,
    'negative_per_positive_ratio': 3,
    'min_negative_per_image': 5
}

postprocess = {
    'score_threshold': .1,
    'max_total': 200,
    'nms': {
        'max_per_class': 100,
        'overlap_threshold': .45
    }
}

target_assigner = {
    'matched_threshold': 0.5,
    'unmatched_threshold': 0.5,
}

augmentations = [
    {'name': 'RandomAdjustHueSaturation', 'args': {'max_hue_delta': .1, 'saturation_delta_range': (.5, 1.5)}},
    {'name': 'ToFloat'},
    {'name': 'RandomAdjustBrightness', 'args': {'max_brightness_delta': .15}},
    {'name': 'RandomAdjustContrast', 'args': {'contrast_delta_range': (.5, 1.5)}},
    {'name': 'RandomExpand'},
    {
        'name': 'OneOf',
        'args': {
            'transforms': [
                {'name': 'Identity'},
                {'name': 'RandomCrop', 'args': {'min_iou': .0}},
                {'name': 'RandomCrop', 'args': {'min_iou': .1}},
                {'name': 'RandomCrop', 'args': {'min_iou': .3}},
                {'name': 'RandomCrop', 'args': {'min_iou': .5}},
                {'name': 'RandomCrop', 'args': {'min_iou': .7}},
                {'name': 'RandomCrop', 'args': {'min_iou': .9}},
            ]
        }
    },
    {'name': 'RandomHorizontalFlip'}
]

preprocessing = [
    {'name': 'ToFloatTensor', 'args': {'normalize': True}},
    {'name': 'Normalize', 'args': {'mean': 128 / 255, 'std': 128 / 255}},
    # {'name': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
]

input_size = (300, 300)

# dataset = {
#     'train': {
#         'name': 'Voc',
#         'root': '/home/george.mironov/documents/voc',
#         'image_sets': [(2007, 'trainval'), (2012, 'trainval')]
#     },
#     'val': {
#         'name': 'Voc',
#         'root': '/home/george.mironov/documents/voc',
#         'image_sets': [(2007, 'test')]
#     }
# }
dataset = {
    'train': {
        'name': 'Coco',
        'root': '/home/george.mironov/documents/coco2017'
    },
    'val': {
        'name': 'Coco',
        'root': '/home/george.mironov/documents/coco2017',
        'val': True
    }
}

batch_size = 32
shuffle = True
num_workers = 4

train = {
    'accumulation_steps': 1,
    'epochs': 500,
    'eval_every': 10,

    'optimizer': {
        'name': 'SGD',
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4
    },

    'scheduler': {
        'name': 'MultiStepLR',
        'milestones': [120, 160],
        'gamma': 0.1
    }

    # 'scheduler': {
    #     'name': 'CosineAnnealingLR',
    #     'run_each_step': True,
    #     'T_max': '{total_train_steps} * {epochs}',
    #     'eta_min': 0
    # }
}
