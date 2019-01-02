import glob
import os
from xml.etree import ElementTree

import numpy as np
import tqdm

from bf.datasets.detection_dataset import DetectionDataset
from bf.utils import xml_utils


class CustomVoc(DetectionDataset):
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

        for annotation in tqdm.tqdm(glob.glob(os.path.join(root, '**', '*.xml'), recursive=True), desc=root):
            xmldict = xml_utils.XmlDictConfig(ElementTree.parse(annotation).getroot())

            width = int(xmldict['size']['width'])
            height = int(xmldict['size']['height'])
            objects = xmldict.get('object', [])
            objects = objects if isinstance(objects, list) else [objects]

            boxes = []
            for x in objects:
                label = x['name'].lower()
                if label in label_map:
                    label = label_map[label]
                if label != 'background':
                    boxes.append([
                        max(int(x['bndbox']['xmin']), 0),
                        max(int(x['bndbox']['ymin']), 0),
                        min(int(x['bndbox']['xmax']), width - 1),
                        min(int(x['bndbox']['ymax']), height - 1),
                        self.class_labels.index(label),
                        int(x.get('difficult', 0))
                    ])

            self.annotations.append({
                'image_path': annotation.replace('.xml', '.jpg'),
                'width': width,
                'height': height,
                'boxes': np.array(boxes, dtype=np.float32).reshape((-1, 6))
            })
