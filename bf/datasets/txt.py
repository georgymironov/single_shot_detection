import glob
import logging
import os

import numpy as np
import tqdm

from bf.datasets.detection_dataset import DetectionDataset


def _sanity_check(box):
    return box[0] < box[2] and box[1] < box[3]

class Txt(DetectionDataset):
    def __init__(self,
                 root,
                 labels,
                 label_map={},
                 resize=None,
                 augment=None,
                 preprocess=None):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        self.resize = resize
        self.augment = augment
        self.preprocess = preprocess

        self.annotations = []

        for path in tqdm.tqdm(glob.glob(os.path.join(root, '**', '*.txt'), recursive=True), desc=root):
            with open(path, 'r') as f:
                boxes = []
                for line in f.read().splitlines():
                    line = line.split(' ')

                    box = [float(x) for x in line[:4]]

                    if not _sanity_check(box):
                        logging.warn(f'WW Invalid box, skipping: {path}')
                        break

                    if len(line) == 4:
                        line += [1, 1.0]
                        logging.warn(f'WW No class is specified for {path}, assuming {labels[1]}')
                    if len(line) == 5:
                        line += [1.0]

                    label = line[4].lower()
                    if label in label_map:
                        label = label_map[label]
                    if label == 'background':
                        continue
                    label = self.class_labels.index(label)

                    boxes.append(box + [label, line[5]])

                else:
                    self.annotations.append({
                        'image_path': os.path.splitext(path)[0],
                        'boxes': np.array(boxes, dtype=np.float32)
                    })
