from collections import defaultdict
import csv
import logging
import os

import numpy as np

from bf.datasets.detection_dataset import DetectionDataset


def _sanity_check(box):
    return box[0] < box[2] or box[1] < box[3]

class Csv(DetectionDataset):
    def __init__(self,
                 path,
                 labels,
                 label_map={},
                 resize=None,
                 augment=None,
                 preprocess=None,
                 delimiter=','):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        self.resize = resize
        self.augment = augment
        self.preprocess = preprocess

        annotations = defaultdict(list)

        with open(path, 'r') as f:
            logging.info(f'===> Loading {path}')
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                annotations[row[0]].append(row[1:])

        self.annotations = []
        for name, boxes in annotations.items():
            self.annotations.append({
                'image_path': os.path.join(os.path.dirname(path), f'{name}.jpg'),
                'boxes': np.array(boxes, dtype=np.float32)
            })
