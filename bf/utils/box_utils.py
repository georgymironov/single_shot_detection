import functools

import numpy as np
import torch


def to_torch(func):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        if isinstance(args[0], np.ndarray):
            return func(*[torch.from_numpy(x) for x in args], **kwargs).numpy()
        return func(*args, **kwargs)
    return wrapped_function

def to_corners(box):
    """
    Args:
        box: torch.tensor(:shape [...Boxes, 4])
    Returns:
        minmax: torch.tensor(:shape [...Boxes, 4])
    """
    return torch.cat([box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2], dim=-1)

def to_centroids(box):
    """
    Args:
        box: torch.tensor(:shape [...Boxes, 4])
    Returns:
        centroids: torch.tensor(:shape [...Boxes, 4])
    """
    return torch.cat([(box[..., 2:] + box[..., :2]) / 2, box[..., 2:] - box[..., :2]], dim=-1)

def area(box):
    """
    Note: Boxes should have minmax format
    Args:
        box: torch.tensor(:shape [...Boxes, 4])
    Returns:
        area: torch.tensor(:shape [...Boxes, 4])
    """
    return (box[..., 2] - box[..., 0]).clamp_(0) * (box[..., 3] - box[..., 1]).clamp_(0)

@to_torch
def intersection(a, b, cartesian=True, zero_incorrect=False):
    """
    Note: Boxes should have minmax format
    Args:
        a: torch.tensor(:shape [BoxesA, 4])
        b: torch.tensor(:shape [BoxesB, 4])
        cartesian: bool
        zero_incorrect: bool
    Returns:
        intersection: torch.tensor(:shape [BoxesA, BoxesB])
    """
    if cartesian:
        min_ = torch.max(
            a[:, :2].unsqueeze(1).expand((a.size(0), b.size(0), 2)),
            b[:, :2].unsqueeze(0).expand((a.size(0), b.size(0), 2))
        )
        max_ = torch.min(
            a[:, 2:].unsqueeze(1).expand((a.size(0), b.size(0), 2)),
            b[:, 2:].unsqueeze(0).expand((a.size(0), b.size(0), 2))
        )
    else:
        assert a.size() == b.size()
        min_ = torch.max(a[..., :2], b[..., :2])
        max_ = torch.min(a[..., 2:], b[..., 2:])

    intersection = torch.cat([min_, max_], dim=-1)

    if zero_incorrect:
        negative = (max_ < min_).any(dim=-1)
        intersection[negative] = 0

    return intersection

@to_torch
def jaccard(a, b, cartesian=True):
    """
    Note: Boxes should have minmax format
    Args:
        a: torch.tensor(:shape [BoxesA, 4])
        b: torch.tensor(:shape [BoxesB, 4])
        cartesian: bool
    Returns:
        jaccard: torch.tensor(:shape [BoxesA, BoxesB])
    """
    intersection_area = area(intersection(a, b, cartesian))
    area_a = area(a)
    area_b = area(b)

    if cartesian:
        area_a = area_a.unsqueeze(1).expand_as(intersection_area)
        area_b = area_b.unsqueeze(0).expand_as(intersection_area)

    return intersection_area / (area_a + area_b - intersection_area)

@to_torch
def nms(boxes, scores, overlap_threshold, max_per_class=None):
    """
    Note: Boxes should have minmax format
    Args:
        boxes: torch.tensor(:shape [Boxes, 4])
        scores: torch.tensor(:shape [Boxes])
        overlap_threshold: float
        max_per_class: int
    Returns:
        picked: tuple(
                    tuple(torch.tensor(:shape [Picked, 4]) -> boxes_picked,
                          torch.tensor(:shape [Picked]) -> scores_picked)
                    torch.tensor(:shape [Picked]) -> indexes_picked
                )
    """
    if max_per_class is not None and max_per_class < scores.size(0):
        scores, indexes = scores.topk(max_per_class, sorted=True, largest=True)
        boxes = boxes[indexes]
        indexes = torch.arange(boxes.size(0), dtype=torch.long)
    else:
        _, indexes = scores.sort(descending=True)

    box_area = area(boxes)
    picked = []

    while indexes.size(0) > 0:
        i = indexes[0]
        picked.append(i)
        indexes = indexes[1:]

        intersection_area = area(intersection(boxes[i].unsqueeze(0), boxes[indexes]).squeeze_(0))
        iou = intersection_area / (box_area[i] + box_area[indexes] - intersection_area)
        indexes = indexes[iou < overlap_threshold]

    picked = torch.tensor(picked, dtype=torch.long)
    return (boxes[picked], scores[picked]), picked
