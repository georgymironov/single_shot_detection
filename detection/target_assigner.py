import torch

from bf.utils import box_utils
from bf.datasets import detection_dataset as det_ds
from detection.matcher import match_per_prediction


LOC_INDEX_START = 0
LOC_INDEX_END = 4
CLASS_INDEX = 4
SCORE_INDEX = 5

IGNORE_INDEX = -1


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

        target = torch.zeros((batch_size, num_anchors, 6), dtype=torch.float32, device=device)
        target[..., SCORE_INDEX] = torch.ones_like(target[..., SCORE_INDEX])

        for i, gt in enumerate(ground_truth):
            if not len(gt):
                continue

            gt_boxes = gt[:, det_ds.LOC_INDEX_START:det_ds.LOC_INDEX_END]
            weights = box_utils.jaccard(gt_boxes, corner_anchors)

            box_idx = match_per_prediction(weights, self.matched_threshold, self.unmatched_threshold)
            matched = box_idx >= 0

            target[i, matched, LOC_INDEX_START:LOC_INDEX_END] = gt[box_idx[matched], det_ds.LOC_INDEX_START:det_ds.LOC_INDEX_END]
            target[i, matched, CLASS_INDEX] = gt[box_idx[matched], det_ds.CLASS_INDEX]
            target[i, matched, SCORE_INDEX] = gt[box_idx[matched], det_ds.SCORE_INDEX]

            ingored = box_idx.eq(IGNORE_INDEX)
            target[i, ingored, CLASS_INDEX] = IGNORE_INDEX
            target[i, ingored, SCORE_INDEX] = IGNORE_INDEX

        box_utils.to_centroids(target[..., LOC_INDEX_START:LOC_INDEX_END], inplace=True)
        self.box_coder.encode_box(target[..., LOC_INDEX_START:LOC_INDEX_END], anchors, inplace=True)

        assert not torch.isnan(target[..., LOC_INDEX_START:LOC_INDEX_END][target[..., CLASS_INDEX].gt(0)]).any().item()

        return target
