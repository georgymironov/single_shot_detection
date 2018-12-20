import functools
import math

import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    def __init__(self,
                 aspect_ratios,
                 min_scale=None,
                 max_scale=None,
                 min_size=None,
                 max_size=None,
                 step=None,
                 offset=[.5, .5],
                 num_branches=1,
                 clip=False):
        super(AnchorGenerator, self).__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.num_ratios = (len(aspect_ratios) + 1)
        self.num_branches = num_branches
        self.num_boxes = self.num_ratios * num_branches
        self.clip = clip
        self.offset = offset
        self.step = step

    @functools.lru_cache()
    def _generate_anchors(self, img_size, feature_map_size):
        img_w, img_h = img_size
        layer_w, layer_h = feature_map_size
        device = 'cpu'

        boxes = torch.empty((layer_h, layer_w, self.num_boxes, 4), dtype=torch.float32, device=device)

        if self.step is not None:
            step_w = self.step
            step_h = self.step
        else:
            step_w = img_w / layer_w
            step_h = img_h / layer_h

        hws = torch.empty((self.num_boxes, 2), dtype=torch.float32, device=device)

        if self.min_size is not None and self.max_size is not None:
            sizes = torch.linspace(self.min_size, self.max_size, self.num_branches + 1).unsqueeze(1).expand(-1, 2)
        else:
            scales = torch.linspace(self.min_scale, self.max_scale, self.num_branches + 1).unsqueeze(1)
            sizes = torch.cat([scales * img_w, scales * img_h], dim=1)

        for j in range(self.num_branches):
            min_size = sizes[j]
            max_size = sizes[j + 1]

            for i, r in enumerate(self.aspect_ratios):
                hws[j * self.num_ratios + i][0] = min_size[0] * math.sqrt(r)
                hws[j * self.num_ratios + i][1] = min_size[1] / math.sqrt(r)

            hws[j * self.num_ratios + i + 1][0] = math.sqrt(min_size[0] * max_size[0])
            hws[j * self.num_ratios + i + 1][1] = math.sqrt(min_size[1] * max_size[1])

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

    def forward(self, img, feature_map):
        """
        Args:
            img: torch.tensor(:shape [Batch, Channels, Height, Width])
            feature_map: torch.tensor(:shape [Batch, Channels, Height, Width])
        Returns:
            priors: torch.tensor(:shape [Height, Width, AspectRatios, 4])
        """
        img_size = img.size(3), img.size(2)
        feature_map_size = feature_map.size(3), feature_map.size(2)

        anchors = self._generate_anchors(img_size, feature_map_size)

        return anchors
