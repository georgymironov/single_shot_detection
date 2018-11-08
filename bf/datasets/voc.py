import os
from xml.etree import ElementTree

from jpeg4py import JPEG
import numpy as np
import torch
from torch.utils.data import Dataset

from bf.utils import dataset_utils, xml_utils


class Voc(Dataset):
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
                 resize=None,
                 augment=None,
                 preprocess=None):
        self.resize = resize
        self.augment = augment
        self.preprocess = preprocess

        self.annotations = []

        for year, image_set in image_sets:
            image_set_file = os.path.join(root, f'VOC{year}', 'ImageSets', 'Main', f'{image_set}.txt')

            with open(image_set_file, 'r') as f:
                print(f'===> Loading {image_set_file}')
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
                    int(x['difficult'])
                ] for x in objects]

                self.annotations.append({
                    'image_path': os.path.join(root, f'VOC{year}', 'JPEGImages', xmldict['filename']),
                    'width': width,
                    'height': height,
                    'boxes': np.array(boxes, dtype=np.float32)
                })

        print(f'===> Pascal VOC {image_sets} loaded. {len(self)} images total')

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
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        targets = list(targets)

        return imgs, targets

    def display(self, index):
        dataset_utils.display(*self[index])
