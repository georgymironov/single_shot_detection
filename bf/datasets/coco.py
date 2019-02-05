from collections import defaultdict
import logging
import json
import os

import numpy as np

from bf.datasets.detection_dataset import DetectionDataset


class Coco(DetectionDataset):
    class_labels = ('background',
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush')
    num_classes = len(class_labels)

    def __init__(self,
                 root,
                 year=2017,
                 val=False,
                 with_crowd=True,
                 resize=None,
                 augment=None,
                 preprocess=None):
        super(Coco, self).__init__()

        self.resize = resize
        self.augment = augment
        self.preprocess = preprocess

        folder = 'val' if val else 'train'
        annotations = os.path.join(root, f'annotations/instances_{folder}{year}.json')
        img_dir = os.path.join(root, f'{folder}{year}')

        with open(annotations, 'r') as f:
            logging.info(f'===> Loading {annotations}')
            annotations = json.load(f)

        images = {x['id']: x for x in annotations['images']}
        self.annotations = defaultdict(lambda: {'boxes': []})

        categories = {x['id']: self.class_labels.index(x['name']) for x in annotations['categories']}

        for a in annotations['annotations']:
            image = images[a['image_id']]
            self.annotations[a['image_id']]['image_path'] = os.path.join(img_dir, image['file_name'])
            self.annotations[a['image_id']]['width'] = image['width']
            self.annotations[a['image_id']]['height'] = image['height']
            self.annotations[a['image_id']]['boxes'].append(a['bbox'] + [categories[a['category_id']]])
        self.annotations = list(self.annotations.values())

        self._fix_boxes()

        logging.info(f'===> COCO {folder.capitalize()} {year} loaded. {len(self)} images total')

    def _fix_boxes(self):
        for a in self.annotations:
            boxes = []
            for box in a['boxes']:
                if box[2] > 1 and box[3] > 1:
                    boxes.append([
                        max(box[0], 0.),
                        max(box[1], 0.),
                        min(box[0] + box[2], a['width'] - 1.),
                        min(box[1] + box[3], a['height'] - 1.),
                        box[4]
                    ])
            a['boxes'] = np.array(boxes, dtype=np.float32)
