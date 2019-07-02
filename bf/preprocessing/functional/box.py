import cv2
import numpy as np

from bf.utils import box_utils


class box(object):
    @staticmethod
    def resize(target, new_w, new_h, width, height):
        target = target.copy()
        target[:, [0, 2]] *= new_w / width
        target[:, [1, 3]] *= new_h / height

        target[..., 0].clip(min=0, max=new_w - 1, out=target[..., 0])
        target[..., 1].clip(min=0, max=new_h - 1, out=target[..., 1])
        target[..., 2].clip(min=0, max=new_w - 1, out=target[..., 2])
        target[..., 3].clip(min=0, max=new_h - 1, out=target[..., 3])

        return target

    @staticmethod
    def horizontal_flip(target, width):
        target = target.copy()
        target[..., [0, 2]] = width - 1 - target[..., [2, 0]]
        return target

    @staticmethod
    def vertical_flip(target, height):
        target = target.copy()
        target[..., [1, 3]] = height - 1 - target[..., [3, 1]]
        return target

    @staticmethod
    def rotate(target, width, height, angle):
        assert width == height

        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        def _rotate(point):
            homogenous = np.ones((point.shape[0], 3), dtype=point.dtype)
            homogenous[:, :2] = point
            return homogenous.dot(M.T)

        target = target.copy()

        a = _rotate(target[:, :2])
        b = _rotate(target[:, 2:4])

        target[:, 0] = np.min([a[..., 0], b[..., 0]], axis=0)
        target[:, 1] = np.min([a[..., 1], b[..., 1]], axis=0)
        target[:, 2] = np.max([a[..., 0], b[..., 0]], axis=0)
        target[:, 3] = np.max([a[..., 1], b[..., 1]], axis=0)

        target[..., 0].clip(min=0, max=width - 1, out=target[..., 0])
        target[..., 1].clip(min=0, max=height - 1, out=target[..., 1])
        target[..., 2].clip(min=0, max=width - 1, out=target[..., 2])
        target[..., 3].clip(min=0, max=height - 1, out=target[..., 3])

        return target

    @staticmethod
    def crop(target, xmin, ymin, width, height, min_iou=0.5, keep_criterion='center_point', min_objects_kept=1):
        if len(target) == 0:
            return target

        region = np.array([xmin, ymin, xmin + width - 1, ymin + height - 1], dtype=np.float32)
        new_target = target.copy()
        new_target[:, :4] = box_utils.intersection(region[np.newaxis], target[:, :4], zero_incorrect=True).squeeze()
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
                return None

            new_target[..., [0, 2]] -= xmin
            new_target[..., [1, 3]] -= ymin
            new_target[..., 0].clip(min=0, max=width - 1, out=new_target[..., 0])
            new_target[..., 1].clip(min=0, max=height - 1, out=new_target[..., 1])
            new_target[..., 2].clip(min=0, max=width - 1, out=new_target[..., 2])
            new_target[..., 3].clip(min=0, max=height - 1, out=new_target[..., 3])

            return new_target

    @staticmethod
    def expand(target, xmin, ymin, width=None, height=None):
        target = target.copy()
        target[..., [0, 2]] += xmin
        target[..., [1, 3]] += ymin
        return target
