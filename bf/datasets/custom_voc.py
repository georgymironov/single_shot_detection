import glob
import os
from xml.etree import ElementTree

from jpeg4py import JPEG
import numpy as np
from torch.utils.data import Dataset

from bf.utils import dataset_utils, xml_utils


class CustomVoc(Dataset):
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

        for annotation in glob.glob(os.path.join(root, '*.xml')):
            xmldict = xml_utils.XmlDictConfig(ElementTree.parse(annotation).getroot())

            width = int(xmldict['size']['width'])
            height = int(xmldict['size']['height'])
            objects = xmldict['object'] if isinstance(xmldict['object'], list) else [xmldict['object']]

            boxes = [[
                max(int(x['bndbox']['xmin']), 0),
                max(int(x['bndbox']['ymin']), 0),
                min(int(x['bndbox']['xmax']), width - 1),
                min(int(x['bndbox']['ymax']), height - 1),
                self.class_labels.index(x['name']),
                int(x.get('difficult', 0))
            ] for x in objects]

            self.annotations.append({
                'image_path': os.path.join(root, xmldict['filename']),
                'width': width,
                'height': height,
                'boxes': np.array(boxes, dtype=np.float32)
            })

        print(f'===> {root} loaded. {len(self)} images total')

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

    @staticmethod
    def collate(batch):
        return dataset_utils.collate_detections(batch)

    def display(self, index):
        dataset_utils.display(*self[index])
