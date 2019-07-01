import logging
import os
from xml.etree import ElementTree

import numpy as np

from bf.datasets.detection_dataset import DetectionDataset
from bf.utils import xml_utils


class Voc(DetectionDataset):
    class_labels = ('background',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(class_labels)

    def __init__(self,
                 root,
                 image_sets,
                 augment=None,
                 preprocess=None):
        self.augment = augment
        self.preprocess = preprocess

        self.annotations = []

        for year, image_set in image_sets:
            image_set_file = os.path.join(root, f'VOC{year}', 'ImageSets', 'Main', f'{image_set}.txt')

            with open(image_set_file, 'r') as f:
                logging.info(f'===> Loading {image_set_file}')
                annotations = [x.strip() for x in f.readlines()]

            for annotation in annotations:
                annotation_file = os.path.join(root, f'VOC{year}', 'Annotations', f'{annotation}.xml')
                xmldict = xml_utils.XmlDictConfig(ElementTree.parse(annotation_file).getroot())

                width = int(xmldict['size']['width'])
                height = int(xmldict['size']['height'])
                objects = xmldict['object'] if isinstance(xmldict['object'], list) else [xmldict['object']]

                boxes = [[
                    max(int(x['bndbox']['xmin']), 0),
                    max(int(x['bndbox']['ymin']), 0),
                    min(int(x['bndbox']['xmax']), width - 1),
                    min(int(x['bndbox']['ymax']), height - 1),
                    self.class_labels.index(x['name']),
                    1.0,
                    int(x['difficult'])
                ] for x in objects]

                self.annotations.append({
                    'image_path': os.path.join(root, f'VOC{year}', 'JPEGImages', xmldict['filename']),
                    'width': width,
                    'height': height,
                    'boxes': np.array(boxes, dtype=np.float32)
                })

        logging.info(f'===> Pascal VOC {image_sets} loaded. {len(self)} images total')
