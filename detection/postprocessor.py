import functools

import torch

from bf.utils import box_utils


class Postprocessor(object):
    def __init__(self, box_coder, score_threshold, nms, max_total=None):
        self.box_coder = box_coder
        self.score_threshold = score_threshold
        self.nms = functools.partial(box_utils.nms, **nms)
        self.max_total = max_total

    def postprocess(self, prediction, priors):
        b_scores, b_boxes = prediction
        num_classes = b_scores.size(-1)
        assert b_scores.dim() == 3

        b_scores.exp_()

        b_scores = b_scores.cpu()
        b_boxes = b_boxes.cpu()

        b_boxes = self.box_coder.decode_box(b_boxes, priors, inplace=False)
        b_boxes = box_utils.to_corners(b_boxes)

        processed = []
        for scores, boxes in zip(b_scores, b_boxes):
            picked = []

            for class_index in range(1, num_classes):
                class_scores = scores[:, class_index]
                mask = class_scores > self.score_threshold

                (boxes_picked, scores_picked), _ = self.nms(boxes[mask], class_scores[mask])
                classes_picked = torch.full_like(scores_picked.unsqueeze_(1), class_index, dtype=torch.float)

                picked.append(torch.cat([boxes_picked, scores_picked, classes_picked], dim=-1))

            picked = torch.cat(picked, dim=0)

            if self.max_total is not None and self.max_total < picked.size(0):
                _, indexes = torch.topk(picked[:, 4], self.max_total, sorted=True, largest=True)
                picked = picked[indexes]

            processed.append(picked)

        return processed
