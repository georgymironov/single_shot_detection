import functools

import torch
import torch.nn.functional as F

from bf.utils import box_utils


class Postprocessor(object):
    def __init__(self, box_coder, score_threshold, nms, score_converter='SOFTMAX', max_total=None):
        self.box_coder = box_coder
        self.score_threshold = score_threshold
        self.nms = functools.partial(box_utils.nms, score_threshold=score_threshold, **nms)
        self.max_total = max_total
        self.score_converter = score_converter

        if score_converter == 'SIGMOID':
            self.score_converter_fn = torch.sigmoid
        elif score_converter == 'SOFTMAX':
            self.score_converter_fn = functools.partial(F.softmax, dim=-1)
        else:
            raise ValueError(f'Wrong value for score_converter: {score_converter}')

    def postprocess(self, prediction, priors):
        """
        Args:
            prediction: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes * Classes])
                torch.tensor(:shape [Batch, AnchorBoxes * 4])
            priors: torch.tensor(:shape [AnchorBoxes, 4]
        Returns:
            processed: list(:len Batch) of torch.tensor(:shape [Boxes_i, 6] ~ {[0-3] - box, [4] - class, [5] - score})
        """
        b_scores, b_boxes = prediction

        batch_size = b_scores.size(0)
        num_priors = priors.size(0)

        b_scores = b_scores.view(batch_size, num_priors, -1)
        b_scores = self.score_converter_fn(b_scores)
        num_classes = b_scores.size(-1)

        if self.score_converter == 'SOFTMAX':
            num_classes = num_classes - 1
            b_scores = b_scores[..., 1:]

        b_scores = b_scores.cpu()

        b_boxes = b_boxes.view(batch_size, num_priors, 4)
        b_boxes = b_boxes.to(priors.device)
        b_boxes = self.box_coder.decode_box(b_boxes, priors, inplace=False)
        b_boxes = box_utils.to_corners(b_boxes)
        b_boxes = b_boxes.cpu()

        processed = []
        for scores, boxes in zip(b_scores, b_boxes):
            picked = []

            for class_index in range(0, num_classes):
                class_scores = scores[:, class_index]
                mask = class_scores > self.score_threshold

                (boxes_picked, scores_picked), _ = self.nms(boxes[mask], class_scores[mask])
                classes_picked = torch.full_like(scores_picked.unsqueeze_(1), class_index + 1, dtype=torch.float)

                picked.append(torch.cat([boxes_picked, classes_picked, scores_picked], dim=-1))

            picked = torch.cat(picked, dim=0)

            if self.max_total is not None and self.max_total < picked.size(0):
                _, indexes = torch.topk(picked[:, 5], self.max_total, sorted=True, largest=True)
                picked = picked[indexes]

            processed.append(picked)

        return processed
