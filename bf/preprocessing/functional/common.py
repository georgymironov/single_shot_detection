import math
import random

import cv2
import numpy as np


def resize(sample, size, target_fn=None, interpolation=cv2.INTER_LINEAR):
    img, target = sample
    h, w = img.shape[:2]
    new_w, new_h = size

    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    if target_fn is not None:
        target = target_fn(target, new_w, new_h, w, h)

    return img, target

def horizontal_flip(sample, target_fn=None):
    img, target = sample
    img = np.fliplr(img)

    if target_fn is not None:
        target = target_fn(target, img.shape[1])

    return img, target

def vertical_flip(sample, target_fn=None):
    img, target = sample
    img = np.flipud(img)

    if target_fn is not None:
        target = target_fn(target, img.shape[0])

    return img, target

def random_rotate(sample, target_fn=None):
    img, target = sample
    height, width = img.shape[:2]
    assert height == width

    angle = random.randrange(4) * 90

    if angle == 0:
        return img, target

    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (width, height))

    if target_fn is not None:
        target = target_fn(target, width, height, angle)

    return img, target

def random_crop(sample,
                target_fn=None,
                aspect_ratio_range=(0.5, 2.),
                area_range=(0.1, 1.),
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

        if target_fn is not None:
            new_target = target_fn(target, xmin, ymin, new_w, new_h)
        else:
            new_target = target

        if new_target is not None:
            return img[ymin:ymin + new_h, xmin:xmin + new_w], new_target

    return img, target

def random_expand(sample,
                  target_fn=None,
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

        if target_fn is not None:
            target = target_fn(target, xmin, ymin, new_w, new_h)

        return new_img, target

    return img, target
