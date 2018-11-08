import torch


class BoxCoder(object):
    def __init__(self, xy_scale, wh_scale, eps=1e-8):
        self.xy_scale = xy_scale
        self.wh_scale = wh_scale
        self.eps = eps

    def encode_box(self, boxes, priors, inplace=False):
        """
        Args:
            boxes: torch.tensor(:shape [Batch, AnchorBoxes, 4])
            priors: torch.tensor(:shape [AnchorBoxes, 4])
        Returns:
            encoded: torch.tensor(:shape [Batch, AnchorBoxes, 4])
        """
        priors = priors.unsqueeze(0)
        if inplace:
            boxes[..., :2] -= priors[..., :2]
            boxes[..., :2] /= priors[..., 2:]
            boxes[..., :2] *= self.xy_scale
            boxes[..., 2:] /= priors[..., 2:]
            boxes[..., 2:] += self.eps
            boxes[..., 2:].log_()
            boxes[..., 2:] *= self.wh_scale
        else:
            return torch.cat([
                (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] * self.xy_scale,
                torch.log((boxes[..., 2:] + self.eps) / priors[..., 2:]) * self.wh_scale], dim=-1)

    def decode_box(self, boxes, priors, inplace=False):
        """
        Args:
            boxes: torch.tensor(:shape [Batch, AnchorBoxes, 4])
            priors: torch.tensor(:shape [AnchorBoxes, 4])
        Returns:
            decoded: torch.tensor(:shape [Batch, AnchorBoxes, 4])
        """
        priors = priors.unsqueeze(0)
        if inplace:
            boxes[..., :2] /= self.xy_scale
            boxes[..., :2] *= priors[..., 2:]
            boxes[..., :2] += priors[..., :2]
            boxes[..., 2:] /= self.wh_scale
            boxes[..., 2:].exp_()
            boxes[..., 2:] *= priors[..., 2:]
        else:
            return torch.cat([
                priors[..., :2] + priors[..., 2:] * boxes[..., :2] / self.xy_scale,
                priors[..., 2:] * torch.exp(boxes[..., 2:] / self.wh_scale)], dim=-1)
