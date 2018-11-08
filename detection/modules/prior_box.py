import math

import torch
import torch.nn as nn


class PriorBox(nn.Module):
    def __init__(self,
                 aspect_ratios,
                 scale,
                 next_scale,
                 step=None,
                 offset=[.5, .5],
                 num_branches=1,
                 clip=False):
        super(PriorBox, self).__init__()

        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.num_ratios = (len(aspect_ratios) + 1)
        self.num_branches = num_branches
        self.num_boxes = self.num_ratios * num_branches
        self.clip = clip
        self.offset = offset
        self.step = step

    def forward(self, img, feature_map):
        """
        Args:
            img: torch.tensor(:shape [Batch, Channels, Height, Width])
            feature_map: torch.tensor(:shape [Batch, Channels, Height, Width])
        Returns:
            priors: torch.tensor(:shape [Height, Width, AspectRatios, 4])
        """
        img_h, img_w = img.size()[2:]
        layer_h, layer_w = feature_map.size()[2:]
        device = 'cpu'

        boxes = torch.empty((layer_h, layer_w, self.num_boxes, 4), dtype=torch.float32, device=device)

        if self.step is not None:
            step_w = self.step
            step_h = self.step
        else:
            step_w = img_w / layer_w
            step_h = img_h / layer_h

        hws = torch.empty((self.num_boxes, 2), dtype=torch.float32, device=device)
        scales = torch.linspace(self.scale, self.next_scale, self.num_branches + 1)

        for j in range(self.num_branches):
            scale = scales[j]
            next_scale = scales[j + 1]

            for i, r in enumerate(self.aspect_ratios):
                hws[j * self.num_ratios + i][0] = img_w * scale * math.sqrt(r)
                hws[j * self.num_ratios + i][1] = img_h * scale / math.sqrt(r)

            hws[j * self.num_ratios + i + 1][0] = img_w * math.sqrt(scale * next_scale)
            hws[j * self.num_ratios + i + 1][1] = img_h * math.sqrt(scale * next_scale)

        xs = torch.linspace(self.offset[0] * step_w, (self.offset[0] + layer_w - 1) * step_w, layer_w, device=device)
        ys = torch.linspace(self.offset[1] * step_h, (self.offset[1] + layer_h - 1) * step_h, layer_h, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs)

        boxes[..., 0] = x_grid.unsqueeze_(-1)
        boxes[..., 1] = y_grid.unsqueeze_(-1)
        boxes[..., 2] = hws[..., 0]
        boxes[..., 3] = hws[..., 1]

        if self.clip:
            boxes[..., [0, 2]].clamp_(0, img_w - 1)
            boxes[..., [1, 3]].clamp_(0, img_h - 1)

        return boxes
