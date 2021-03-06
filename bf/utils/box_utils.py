import functools

import numpy as np
import torch
import torchvision


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

def to_centroids(box, inplace=False):
    """
    Args:
        box: torch.tensor(:shape [...Boxes, 4])
    Returns:
        centroids: torch.tensor(:shape [...Boxes, 4])
    """
    if inplace:
        box[..., 2:] -= box[..., :2]
        box[..., :2] += box[..., 2:] / 2
    else:
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
def iou(a, b, cartesian=True):
    """
    Note: Boxes should have minmax format
    Args:
        a: torch.tensor(:shape [BoxesA, 4])
        b: torch.tensor(:shape [BoxesB, 4])
        cartesian: bool
    Returns:
        iou: torch.tensor(:shape [BoxesA, BoxesB])
    """
    intersection_area = area(intersection(a, b, cartesian=cartesian))
    area_a = area(a)
    area_b = area(b)

    if cartesian:
        area_a = area_a.unsqueeze(1).expand_as(intersection_area)
        area_b = area_b.unsqueeze(0).expand_as(intersection_area)

    return intersection_area / (area_a + area_b - intersection_area)

# ref: https://arxiv.org/pdf/1902.09630v2.pdf
def generalized_iou(a, b, cartesian=True):
    """
    Note: Boxes should have minmax format
    Args:
        a: torch.tensor(:shape [BoxesA, 4])
        b: torch.tensor(:shape [BoxesB, 4])
        cartesian: bool
    Returns:
        iou: torch.tensor(:shape [BoxesA, BoxesB])
    """
    assert a.dim() == b.dim() == 2
    assert a.size(1) == b.size(1) == 4

    intersection_area = area(intersection(a, b, cartesian=cartesian))
    area_a = area(a)
    area_b = area(b)

    if cartesian:
        area_a = area_a.unsqueeze(1).expand_as(intersection_area)
        area_b = area_b.unsqueeze(0).expand_as(intersection_area)

    union_area = area_a + area_b - intersection_area

    if cartesian:
        min_ = torch.min(
            a[:, :2].unsqueeze(1).expand((a.size(0), b.size(0), 2)),
            b[:, :2].unsqueeze(0).expand((a.size(0), b.size(0), 2))
        )
        max_ = torch.max(
            a[:, 2:].unsqueeze(1).expand((a.size(0), b.size(0), 2)),
            b[:, 2:].unsqueeze(0).expand((a.size(0), b.size(0), 2))
        )
    else:
        assert a.size() == b.size()
        min_ = torch.min(a[..., :2], b[..., :2])
        max_ = torch.max(a[..., 2:], b[..., 2:])

    enclosing_area = area(torch.cat([min_, max_], dim=-1))

    return intersection_area / union_area - (enclosing_area - union_area) / enclosing_area

def _soft_nms(boxes, scores, score_threshold, sigma=0.5):
    scores_copy = scores.clone()
    mask = scores > score_threshold
    box_area = area(boxes)
    picked = []

    while mask.nonzero().sum():
        idx = scores_copy.argmax()
        scores_copy[idx] = 0
        picked.append(idx)

        mask = scores_copy > score_threshold

        intersection_area = area(intersection(boxes[idx].unsqueeze(0), boxes[mask]).squeeze_(0))
        iou = intersection_area / (box_area[idx] + box_area[mask] - intersection_area)
        scores_copy[mask] = scores_copy[mask] * iou.pow(2).div_(sigma).neg_().exp_()

    picked = torch.tensor(picked, dtype=torch.long)
    return (boxes[picked], scores[picked]), picked

@to_torch
def nms(boxes, scores, overlap_threshold, score_threshold, max_per_class=None, soft=False, sigma=0.5):
    """
    Note: Boxes should have minmax format
    Args:
        boxes:              torch.tensor(:shape [Boxes, 4])
        scores:             torch.tensor(:shape [Boxes])
        overlap_threshold:  float
        score_threshold:    float
        max_per_class:      int
        soft:               bool
        sigma:              float
    Returns:
        picked: tuple(
                    tuple(
                        torch.tensor(:shape [Picked, 4]) -> boxes_picked,
                        torch.tensor(:shape [Picked]) -> scores_picked
                    )
                    torch.tensor(:shape [Picked]) -> indexes_picked
                )
    """
    if max_per_class is not None and max_per_class < scores.size(0):
        scores, indexes = scores.topk(max_per_class, sorted=False, largest=True)
        boxes = boxes[indexes]

    if soft:
        return _soft_nms(boxes, scores, score_threshold, sigma)
    else:
        picked = torchvision.ops.nms(boxes, scores, overlap_threshold)
        return (boxes[picked], scores[picked]), picked
