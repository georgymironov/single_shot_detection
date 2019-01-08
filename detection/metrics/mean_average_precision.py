from collections import defaultdict

import torch

from bf.utils import box_utils


def mean_average_precision(predictions, gts, class_labels, iou_threshold, voc=False, verbose=True):
    """
    Args:
        predictions: torch.tensor(:shape [NumBoxes, 7] ~ {[0] - image_id, [1-4] - box, [5] - class, [6] - score})
        gts: list(:len NumImages) of torch.tensor(:shape [NumBoxes_i, 5/6] ~ {[0-3] - box, [4] - class, [5]? - is_difficult})
        class_labels: dict(:keys ClassId, :values ClassName)
        iou_threshold: float
        voc: bool
        verbose: bool
    Returns:
        mAP: float
    """
    ignore_difficult = gts[0].size(1) == 6
    total_positive = defaultdict(int)
    gt_grouped = []
    gt_classes = []

    for i, gt in enumerate(gts):
        gt_classes.append(gt[..., 4].long())
        gt_by_class = defaultdict(list)

        for y, class_index in zip(gt, gt_classes[i]):
            class_index = class_index.item()
            gt_by_class[class_index].append(y)
            if not ignore_difficult or y[5].item() == 0:
                total_positive[class_index] += 1

        gt_by_class = {class_index: torch.stack(gt_boxes, dim=0) for class_index, gt_boxes in gt_by_class.items()}
        gt_grouped.append(gt_by_class)

    indexes = predictions[:, 6].argsort(dim=0, descending=True)
    predictions = predictions[indexes]

    true_positive = defaultdict(list)
    false_positive = defaultdict(list)
    matched = defaultdict(lambda: defaultdict(set))

    for pred in predictions:
        id_ = int(pred[0].item())
        class_index = int(pred[5].item())
        box = pred[1:5]

        true_positive[class_index].append(0 if len(true_positive[class_index]) == 0 else true_positive[class_index][-1])
        false_positive[class_index].append(0 if len(false_positive[class_index]) == 0 else false_positive[class_index][-1])

        if class_index not in gt_grouped[id_]:
            false_positive[class_index][-1] += 1
            continue

        jaccard = box_utils.jaccard(box.unsqueeze(0), gt_grouped[id_][class_index][:, :4]).squeeze_(0)
        iou, index = jaccard.max(dim=0)
        if iou.item() > iou_threshold:
            if not ignore_difficult or gt_grouped[id_][class_index][index, 5] == 0:
                if index.item() not in matched[id_][class_index]:
                    true_positive[class_index][-1] += 1
                    matched[id_][class_index].add(index.item())
                else:
                    false_positive[class_index][-1] += 1
        else:
            false_positive[class_index][-1] += 1

    average_precision = {}

    for class_index in total_positive.keys():
        average_precision[class_index] = 0.0

    if verbose:
        print('Mean Average Precision results:')

    for class_index in sorted(true_positive.keys()):
        true_positive[class_index] = torch.tensor(true_positive[class_index], dtype=torch.float32)
        false_positive[class_index] = torch.tensor(false_positive[class_index], dtype=torch.float32)

        precision = true_positive[class_index] / (true_positive[class_index] + false_positive[class_index])
        precision = torch.cat([precision, torch.tensor([0.])])

        for i in reversed(range(1, len(precision))):
            precision[i - 1] = torch.max(precision[i - 1], precision[i])

        recall = true_positive[class_index] / total_positive[class_index]

        if voc:
            recall = torch.cat([recall, torch.tensor([1.])])
            indexes = torch.arange(0, 1.1, .1).unsqueeze_(0).expand((recall.size(0), 11)) \
                .gt(recall.unsqueeze(1).expand((recall.size(0), 11))).sum(dim=0)
            average_precision[class_index] = precision[indexes].mean()
        else:
            recall = torch.cat([torch.tensor([0.]), recall, torch.tensor([1.])])
            average_precision[class_index] = (recall[1:] - recall[:-1]).dot(precision)

        average_precision[class_index] = average_precision[class_index].item()

        if verbose:
            print(f'{class_labels[class_index]}: {average_precision[class_index]:6f}')

    mAP = sum(average_precision.values()) / len(average_precision.values())
    if verbose:
        print(f'Total mean: {mAP:6f}')

    return mAP
