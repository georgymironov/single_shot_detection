import torch

from bf.utils import box_utils
from detection.matcher import match_per_prediction


class TargetAssigner(object):
    def __init__(self, box_coder, matched_threshold, unmatched_threshold):
        self.box_coder = box_coder
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold

    def encode_ground_truth(self, ground_truth, priors):
        """
        Args:
            ground_truth: list(:len Batch) of torch.tensor(:shape [Boxes_i, 5])
            priors: torch.tensor(:shape [AnchorBoxes, 4])
        Returns:
            target_classes: torch.tensor(:shape [Batch, AnchorBoxes])
            target_locs: torch.tensor(:shape [Batch, AnchorBoxes, 4])
        """
        batch_size = len(ground_truth)
        num_priors = priors.size(0)
        device = priors.device

        corner_priors = box_utils.to_corners(priors)

        target_classes = torch.zeros((batch_size, num_priors), dtype=torch.long, device=device)
        target_locs = torch.zeros((batch_size, num_priors, 4), dtype=torch.float32, device=device)

        for i, gt in enumerate(ground_truth):
            weights = box_utils.jaccard(gt[:, :4], corner_priors)

            box_idx = match_per_prediction(weights,
                                           matched_threshold=self.matched_threshold,
                                           unmatched_threshold=self.unmatched_threshold)
            matched = box_idx >= 0
            target_classes[i, matched] = gt[box_idx[matched], 4].long()
            target_locs[i, matched] = gt[box_idx[matched], :4]

            ingored = box_idx == -1
            target_classes[i, ingored] = -1

        target_locs = box_utils.to_centroids(target_locs)
        target_locs = self.box_coder.encode_box(target_locs, priors, inplace=False)

        assert not torch.isnan(target_locs[target_classes.gt(0)]).any().item()

        return target_classes, target_locs
