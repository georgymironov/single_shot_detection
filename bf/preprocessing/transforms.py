from abc import ABC, abstractmethod
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from . import functional


class RandomTransform(ABC):
    def __init__(self, p=.5):
        self.p = p

    @abstractmethod
    def apply(self, sample):
        pass

    def __call__(self, sample):
        if random.random() < self.p:
            return self.apply(sample)
        return sample

class OneOf(object):
    def __init__(self, transforms):
        self.transforms = [globals()[x['name']](**x.get('args', {})) for x in transforms]

    def __call__(self, sample):
        return self.transforms[random.randrange(0, len(self.transforms))](sample)

class Identity(object):
    def __call__(self, sample):
        return sample

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return functional.resize(sample, self.size)

class ToFloat(object):
    def __call__(self, sample):
        return sample[0].astype('float32'), sample[1]

class ToUint8(object):
    def __call__(self, sample):
        return sample[0].astype('uint8'), sample[1]

class RandomCrop(RandomTransform):
    def __init__(self,
                 min_iou=.5,
                 aspect_ratio_range=(0.5, 2.),
                 area_range=(0.1, 1.),
                 keep_criterion='center_point',
                 min_objects_kept=1,
                 **kwargs):
        super(RandomCrop, self).__init__(**kwargs)

        self.min_iou = min_iou
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.keep_criterion = keep_criterion
        self.min_objects_kept = min_objects_kept

    def apply(self, sample):
        img, target = functional.random_crop(sample,
                                             min_iou=self.min_iou,
                                             aspect_ratio_range=self.aspect_ratio_range,
                                             area_range=self.area_range,
                                             keep_criterion=self.keep_criterion,
                                             min_objects_kept=self.min_objects_kept)
        return img, target

class RandomCropResize(RandomTransform):
    def __init__(self,
                 size=None,
                 min_iou=.5,
                 aspect_ratio_range=(0.5, 2.),
                 area_range=(0.1, 1.),
                 keep_criterion='center_point',
                 min_objects_kept=1,
                 **kwargs):
        super(RandomCropResize, self).__init__(**kwargs)

        self.size = size
        self.min_iou = min_iou
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.keep_criterion = keep_criterion
        self.min_objects_kept = min_objects_kept

    def apply(self, sample):
        size = self.size if self.size is not None else (sample[0].shape[1], sample[0].shape[0])
        img, target = functional.random_crop(sample,
                                             min_iou=self.min_iou,
                                             aspect_ratio_range=self.aspect_ratio_range,
                                             area_range=self.area_range,
                                             keep_criterion=self.keep_criterion,
                                             min_objects_kept=self.min_objects_kept)
        if size != img.shape[:2]:
            img, target = functional.resize((img, target), size)
        return img, target

class RandomExpand(RandomTransform):
    def __init__(self,
                 aspect_ratio_range=(0.5, 2.0),
                 area_range=(1.0, 16.0),
                 **kwargs):
        super(RandomExpand, self).__init__(**kwargs)

        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def apply(self, sample):
        return functional.random_expand(sample, self.aspect_ratio_range, self.area_range)

class RandomHorizontalFlip(RandomTransform):
    def __init__(self, **kwargs):
        super(RandomHorizontalFlip, self).__init__(**kwargs)

    def apply(self, sample):
        img, target = sample
        img = np.fliplr(img)
        target[..., [0, 2]] = img.shape[1] - 1 - target[..., [2, 0]]

        return img, target

class RandomVerticalFlip(RandomTransform):
    def __init__(self, **kwargs):
        super(RandomVerticalFlip, self).__init__(**kwargs)

    def apply(self, sample):
        img, target = sample
        img = np.flipud(img)
        target[..., [1, 3]] = img.shape[0] - 1 - target[..., [3, 1]]

        return img, target

class RandomAdjustBrightness(RandomTransform):
    def __init__(self, max_brightness_delta, **kwargs):
        super(RandomAdjustBrightness, self).__init__(**kwargs)
        self.max_brightness_delta = max_brightness_delta

    def apply(self, sample):
        img, target = sample
        assert img.dtype == np.float32
        img += random.uniform(-self.max_brightness_delta, self.max_brightness_delta) * 255.
        np.clip(img, 0., 255., out=img)

        return img, target

class RandomAdjustContrast(RandomTransform):
    def __init__(self, contrast_delta_range, **kwargs):
        super(RandomAdjustContrast, self).__init__(**kwargs)
        self.contrast_delta_range = contrast_delta_range

    def apply(self, sample):
        img, target = sample
        assert img.dtype == np.float32
        mean = img.reshape((-1, 3)).mean(axis=0)
        img = mean + random.uniform(*self.contrast_delta_range) * (img - mean)
        np.clip(img, 0., 255., out=img)

        return img, target

class RandomAdjustHueSaturation(object):
    def __init__(self, max_hue_delta=None, saturation_delta_range=None, p=.5):
        self.p = p
        self.max_hue_delta = max_hue_delta
        self.saturation_delta_range = saturation_delta_range

    def __call__(self, sample):
        adjust_hue = self.max_hue_delta and random.random() < self.p
        adjust_saturation = self.saturation_delta_range and random.random() < self.p

        if not adjust_hue and not adjust_saturation:
            return sample

        img, target = sample
        assert img.dtype == np.uint8

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img)

        if adjust_hue:
            h = h.astype('int16')
            h += int(random.uniform(-self.max_hue_delta, self.max_hue_delta) * 180)
            h = np.where(h < 0, h + 180, h)
            h = np.where(h > 180, h - 180, h)
            h = h.astype('uint8')

        if adjust_saturation:
            s = s.astype('float32')
            s *= random.uniform(*self.saturation_delta_range)
            np.clip(s, 0., 255., out=s)
            s = s.astype('uint8')

        img = cv2.merge((h, s, v))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        return img, target

class ToFloatTensor(object):
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, sample):
        img, target = sample

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.normalize:
            img /= 255.

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        elif isinstance(target, list):
            target = torch.tensor(target, dtype=torch.float32)

        return img, target

class Normalize(object):
    def __init__(self, mean=0.0, std=1.0):
        if isinstance(mean, list):
            mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)

        if isinstance(std, list):
            std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        self.mean = mean
        self.std = std

    def __call__(self, samples):
        img, target = samples
        assert img.dtype in [torch.float32]
        img -= self.mean
        img /= self.std

        return img, target
