import random

import cv2
import numpy as np
import torch


colors = {}

def display(img, target):
    if isinstance(img, torch.Tensor):
        img = img.numpy().transpose((1, 2, 0))
    if img.dtype == 'float32':
        img = (img * 255).astype('uint8')
    img = np.ascontiguousarray(img)

    if isinstance(target, torch.Tensor):
        target = target.numpy()

    for cl in target[:, 4]:
        if cl not in colors:
            colors[cl] = [random.randint(0, 255) for _ in range(3)]

    for box in target:
        xmin = int(max(0, box[0]))
        ymin = int(max(0, box[1]))
        xmax = int(min(img.shape[1] - 1, box[2]))
        ymax = int(min(img.shape[0] - 1, box[3]))
        dots = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for i, current in enumerate(dots):
            next_ = dots[(i + 1) % len(dots)]
            cv2.line(img, current, next_, colors[box[4]])

    cv2.imshow('image', img[..., [2, 1, 0]])
    return cv2.waitKey(0)
