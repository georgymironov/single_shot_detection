import glob
import os
from xml.etree import ElementTree

from jpeg4py import JPEG
import numpy as np
import tqdm

from bf.datasets.detection_dataset import DetectionDataset
from bf.utils import xml_utils


class CustomVoc(DetectionDataset):
    def __init__(self,
                 root,
                 labels,
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

            boxes = [[
                max(int(x['bndbox']['xmin']), 0),
                max(int(x['bndbox']['ymin']), 0),
                min(int(x['bndbox']['xmax']), width - 1),
                min(int(x['bndbox']['ymax']), height - 1),
                self.class_labels.index(x['name']),
                int(x.get('difficult', 0))
            ] for x in objects]

            self.annotations.append({
                'image_path': os.path.join(os.path.dirname(annotation), xmldict['filename']),
                'width': width,
                'height': height,
                'boxes': np.array(boxes, dtype=np.float32).reshape((-1, 6))
            })

    def __getitem__(self, index):
        annotation = self.annotations[index]
        img = JPEG(annotation['image_path']).decode()
        target = annotation['boxes'].copy()

        if self.augment:
            img, target = self.augment((img, target))
        if self.resize:
            img, target = self.resize((img, target))
        if self.preprocess:
            img, target = self.preprocess((img, target))

        return img, target

    def __len__(self):
        return len(self.annotations)
