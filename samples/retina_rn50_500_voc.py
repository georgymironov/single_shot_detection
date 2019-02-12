seed = 23
use_gpu = True

model = {
    'base': {
        'name': 'resnet50',
        'weight': 'torchvision'
    },
    'detector': {
        'num_classes': 21,
        'use_depthwise': False,
        'source_layers': (5, 6, 7),
        'fpn_layers': 5,
        'fpn_channels': 256,
        'predictor': {
            'num_layers': 4,
            'num_channels': 256,
            'kernel_size': 3,
            'activation': {'name': 'ReLU', 'args': {'inplace': True}},
            'initializer': {'name': 'normal_', 'args': {'mean': 0, 'std': 0.01}}
        }
    },
    'anchor_generator': {
        'type': 'retina_net',
        'min_level': 3,
        'max_level': 7,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'scale': 4.0,
        'scales_per_level': 3
    }
}

box_coder = {
    'xy_scale': 10.0,
    'wh_scale': 5.0,
}

sampler = {
    'name': 'naive_sampler',
}

loss = {
    'classification_loss': {'name': 'SoftmaxFocalLoss', 'gamma': 2.0, 'alpha': 0.75},
    'localization_loss': {'name': 'SmoothL1Loss'},
    'classification_weight': 1.0,
    'localization_weight': 1.0
}

postprocess = {
    'score_threshold': .01,
    'max_total': 200,
    'nms': {
        'max_per_class': 100,
        'overlap_threshold': .5
    }
}

target_assigner = {
    'matched_threshold': 0.5,
    'unmatched_threshold': 0.4,
}

augmentations = [
    {
        'name': 'RandomCrop',
        'args': {
            'min_iou': 0.0,
            'aspect_ratio_range': (0.5, 2.),
            'area_range': (0.5, 1.),
            'min_objects_kept': 0
        }
    },
    {'name': 'RandomHorizontalFlip'}
]

preprocessing = [
    {'name': 'ToFloatTensor', 'args': {'normalize': True}},
    {'name': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
]

input_size = (500, 500)

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

batch_size = 16
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
        'weight_decay': 1e-4
    },

    'scheduler': {
        'name': 'MultiStepLR',
        'milestones': [30, 60],
        'gamma': 0.1
    }
}
