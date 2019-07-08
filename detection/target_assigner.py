import torch

from bf.utils import box_utils
from bf.datasets import detection_dataset as det_ds
from detection import matcher

LOC_INDEX_START = 0
LOC_INDEX_END = 4
CLASS_INDEX = 4
SCORE_INDEX = 5
TARGET_SIZE = 6

NEGATIVE_CLASS = det_ds.NEGATIVE_CLASS
IGNORE_CLASS = -1


class TargetAssigner(object):
    def __init__(self, box_coder, matched_threshold, unmatched_threshold):
        self.box_coder = box_coder
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold

    def encode_ground_truth(self, ground_truth, anchors):
        """
        Args:
            ground_truth: list(:len Batch) of torch.tensor(:shape [Boxes_i, 5])
            anchors: torch.tensor(:shape [AnchorBoxes, 4])
        Returns:
            target_classes: torch.tensor(:shape [Batch, AnchorBoxes])
            target_locs: torch.tensor(:shape [Batch, AnchorBoxes, 4])
        """
        batch_size = len(ground_truth)
        num_anchors = anchors.size(0)
        device = anchors.device

        ground_truth = [x.to(device, non_blocking=True) for x in ground_truth]
        corner_anchors = box_utils.to_corners(anchors)

        target = torch.zeros((batch_size, num_anchors, TARGET_SIZE), dtype=torch.float32, device=device)
        target[..., CLASS_INDEX] = torch.full_like(target[..., SCORE_INDEX], NEGATIVE_CLASS)
        target[..., SCORE_INDEX] = torch.ones_like(target[..., SCORE_INDEX])

        for i, gt in enumerate(ground_truth):
            if not len(gt):
                continue

            gt_boxes = gt[:, det_ds.LOC_INDEX_START:det_ds.LOC_INDEX_END]
            weights = box_utils.jaccard(gt_boxes, corner_anchors)

            box_idx = matcher.match_per_prediction(weights, self.matched_threshold, self.unmatched_threshold)
            matched = box_idx.ne(matcher.NOT_MATCHED) & box_idx.ne(matcher.IGNORE)

            target[i, matched, LOC_INDEX_START:LOC_INDEX_END] = gt[box_idx[matched], det_ds.LOC_INDEX_START:det_ds.LOC_INDEX_END]
            target[i, matched, CLASS_INDEX] = gt[box_idx[matched], det_ds.CLASS_INDEX]
            target[i, matched, SCORE_INDEX] = gt[box_idx[matched], det_ds.SCORE_INDEX]

            ingored = box_idx.eq(IGNORE_CLASS)
            target[i, ingored, CLASS_INDEX] = IGNORE_CLASS
            target[i, ingored, SCORE_INDEX] = IGNORE_CLASS

        box_utils.to_centroids(target[..., LOC_INDEX_START:LOC_INDEX_END], inplace=True)
        self.box_coder.encode_box(target[..., LOC_INDEX_START:LOC_INDEX_END], anchors, inplace=True)

        positive = target[..., CLASS_INDEX].ne(NEGATIVE_CLASS) & target[..., CLASS_INDEX].ne(IGNORE_CLASS)
        assert not torch.isnan(target[..., LOC_INDEX_START:LOC_INDEX_END][positive]).any().item()

        return target
