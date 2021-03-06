import glob
import logging
import os
from xml.etree import ElementTree

import numpy as np
import tqdm

from bf.datasets.detection_dataset import DetectionDataset
from bf.training.env import get_out_file
from bf.utils import xml_utils


def _sanity_check(box):
    return box[0] < box[2] and box[1] < box[3]

class CustomVoc(DetectionDataset):
    def __init__(self,
                 root,
                 labels,
                 label_map={},
                 augment=None,
                 preprocess=None):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        self.augment = augment
        self.preprocess = preprocess

        self.annotations = []

        for annotation in tqdm.tqdm(glob.glob(os.path.join(root, '**', '*.xml'), recursive=True), desc=root, file=get_out_file()):
            xmldict = xml_utils.XmlDictConfig(ElementTree.parse(annotation, parser=ElementTree.XMLParser(encoding="utf-8")).getroot())

            width = int(xmldict['size']['width'])
            height = int(xmldict['size']['height'])
            objects = xmldict.get('object', [])
            objects = objects if isinstance(objects, list) else [objects]

            boxes = []
            for x in objects:
                if x['name'] is None:
                    logging.warn(f'WW Missing label, skipping: {annotation}')
                    break

                label = x['name'].lower()
                if label in label_map:
                    label = label_map[label]
                if label == 'background':
                    continue

                box = [
                    max(int(x['bndbox']['xmin']), 0),
                    max(int(x['bndbox']['ymin']), 0),
                    min(int(x['bndbox']['xmax']), width - 1),
                    min(int(x['bndbox']['ymax']), height - 1),
                    self.class_labels.index(label),
                    1.0,
                    int(x.get('difficult', 0))
                ]
                if not _sanity_check(box):
                    logging.warn(f'WW Invalid box, skipping: {annotation}')
                    break
                boxes.append(box)
            else:
                self.annotations.append({
                    'image_path': annotation.replace('.xml', '.jpg'),
                    'width': width,
                    'height': height,
                    'boxes': np.array(boxes, dtype=np.float32).reshape((-1, 7))
                })
