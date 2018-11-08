import math
import random

import cv2
import numpy as np

from bf.utils import box_utils


def resize(sample, size, interpolation=cv2.INTER_LINEAR):
    img, target = sample
    h, w = img.shape[:2]
    new_w, new_h = size

    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    target[:, [0, 2]] *= new_w / w
    target[:, [1, 3]] *= new_h / h

    return img, target

def random_crop(sample,
                min_iou=.5,
                aspect_ratio_range=(0.5, 2.),
                area_range=(0.1, 1.),
                keep_criterion='center_point',
                min_objects_kept=1,
                attempts=50):
    img, target = sample
    h, w = img.shape[:2]

    for attempt in range(attempts):
        aspect_ratio = random.uniform(*aspect_ratio_range)
        area = random.uniform(*area_range) * h * w
        new_w = int(math.sqrt(area * aspect_ratio))
        new_h = int(math.sqrt(area / aspect_ratio))

        if new_w > w or new_h > h:
            continue

        xmin = random.randint(0, w - new_w)
        ymin = random.randint(0, h - new_h)
        region = np.array([xmin, ymin, xmin + new_w, ymin + new_h], dtype=np.float32)
        new_target = np.empty_like(target)
        new_target[:, :4] = box_utils.intersection(region[np.newaxis], target[:, :4], zero_incorrect=True).squeeze()
        new_target[:, 4] = target[:, 4]
        jaccard = box_utils.jaccard(target[:, :4], new_target[:, :4], cartesian=False)

        if jaccard.max() > min_iou:
            if keep_criterion == 'center_point':
                center = (target[..., :2] + target[..., 2:4]) / 2
                new_target = new_target[np.logical_and(center > region[:2], center < region[2:]).all(axis=1)]
            elif keep_criterion == 'iou':
                new_target = new_target[jaccard > min_iou]
            else:
                raise ValueError(f'Wrong value for keep_criterion: {keep_criterion}')

            if len(new_target) < min_objects_kept:
                continue

            new_target[..., [0, 2]] -= xmin
            new_target[..., [1, 3]] -= ymin
            new_target[..., [0, 1]].clip(min=0, out=new_target[..., [0, 1]])
            new_target[..., 2].clip(max=new_w, out=new_target[..., 2])
            new_target[..., 3].clip(max=new_h, out=new_target[..., 3])

            return img[ymin:ymin + new_h, xmin:xmin + new_w], new_target

    return img, target

def random_expand(sample,
                  aspect_ratio_range=(0.5, 2.0),
                  area_range=(1.0, 16.0),
                  attempts=50):
    img, target = sample
    h, w, d = img.shape

    for attempt in range(attempts):
        aspect_ratio = random.uniform(*aspect_ratio_range)
        area = random.uniform(*area_range) * h * w
        new_w = int(math.sqrt(area * aspect_ratio))
        new_h = int(math.sqrt(area / aspect_ratio))

        if new_w < w or new_h < h:
            continue

        xmin = random.randint(0, new_w - w)
        ymin = random.randint(0, new_h - h)

        new_img = np.full((new_h, new_w, d), img.mean(), dtype=img.dtype)
        new_img[ymin:ymin + h, xmin:xmin + w] = img

        target[..., [0, 2]] += xmin
        target[..., [1, 3]] += ymin

        return new_img, target

    return img, target
