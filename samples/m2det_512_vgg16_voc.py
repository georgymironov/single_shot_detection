seed = 23

model = {
    'base': {
        'name': 'torchvision_vgg16_bn',
        'pretrained': True
    },
    'detector': {
        'num_classes': 81,
        'features': {
            'name': 'MultilevelFeaturePyramid',
            'out_layers': (32, 42),
            'last_feature_layer': 42,
            'num_scales': 6,
            'num_tums': 8,
            'base_reduced_channels': [512, 256]
        },
    },
    'anchor_generator': {
        'type': 'ssd',
        'num_scales': 6,
        'min_scale': 0.15,
        'max_scale': 1.05,
        'aspect_ratios': [[1.0, 2.0]] + [[1.0, 2.0, 3.0]] * 3 + [[1.0, 2.0]] * 2
    }
}

box_coder = {
    'xy_scale': 10.0,
    'wh_scale': 5.0,
}

sampler = {
    'name': 'hard_negative_mining',
    'negative_per_positive_ratio': 3,
    'min_negative_per_image': 5
}

loss = {
    'classification_loss': {'name': 'CrossEntropyLoss'},
    'localization_loss': {'name': 'SmoothL1Loss'},
    'classification_weight': 1.0,
    'localization_weight': 1.0
}

postprocess = {
    'score_threshold': .01,
    'max_total': 200,
    'nms': {
        'max_per_class': 100,
        'overlap_threshold': .45
    },
    'score_converter': 'SOFTMAX'
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
    {'name': 'RandomExpand', 'args': {'aspect_ratio_range': (0.5, 2.0), 'area_range': (1.0, 16.0)}},
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
    {'name': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
]

input_size = (512, 512)

dataset = {
    'train': {
        'name': 'Voc',
        'root': '{HOME}/documents/pascal-voc',
        'image_sets': [(2007, 'trainval'), (2012, 'trainval')]
    },
    'eval': {
        'name': 'Voc',
        'root': '{HOME}/documents/pascal-voc',
        'image_sets': [(2007, 'test')]
    }
}

batch_size = 8
shuffle = True
num_workers = 3

train = {
    'epochs': 500,
    'eval_every': 10,

    'optimizer': {
        'name': 'SGD',
        'lr': 2e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4
    },

    'scheduler': {
        'name': 'MultiStepLR',
        'milestones': [90, 110, 130],
        'gamma': 0.1
    }
}
