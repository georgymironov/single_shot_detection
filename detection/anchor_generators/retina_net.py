import functools
import math

import torch

from bf.utils.misc_utils import filter_kwargs
from ._anchor_generator import _AnchorGenerator


@filter_kwargs
def build_anchor_generators(aspect_ratios,
                            min_level,
                            max_level,
                            scale,
                            scales_per_level):
    return [RetinaAnchorGenerator(aspect_ratios, level, scale, scales_per_level) for level in range(min_level, max_level + 1)]

class RetinaAnchorGenerator(_AnchorGenerator):
    def __init__(self,
                 aspect_ratios,
                 level,
                 scale,
                 scales_per_level=1):
        self.aspect_ratios = aspect_ratios
        self.num_boxes = len(aspect_ratios) * scales_per_level
        self.sizes = [scale * (2 ** (level + x / scales_per_level)) for x in range(scales_per_level)]

    @functools.lru_cache()
    def _generate_anchors(self, img_size, feature_map_size, device='cpu'):
        img_w, img_h = img_size
        layer_w, layer_h = feature_map_size

        boxes = torch.empty((layer_h, layer_w, self.num_boxes, 4), dtype=torch.float32, device=device)

        step_w = img_w / layer_w
        step_h = img_h / layer_h

        hws = torch.empty((self.num_boxes, 2), dtype=torch.float32, device=device)

        for j, size in enumerate(self.sizes):
            for i, ar in enumerate(self.aspect_ratios):
                hws[j * len(self.aspect_ratios) + i][0] = size * math.sqrt(ar)
                hws[j * len(self.aspect_ratios) + i][1] = size / math.sqrt(ar)

        xs = torch.linspace(0.5 * step_w, (0.5 + layer_w - 1) * step_w, layer_w, device=device)
        ys = torch.linspace(0.5 * step_h, (0.5 + layer_h - 1) * step_h, layer_h, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs)

        boxes[..., 0] = x_grid.unsqueeze_(-1)
        boxes[..., 1] = y_grid.unsqueeze_(-1)
        boxes[..., 2] = hws[..., 0]
        boxes[..., 3] = hws[..., 1]

        return boxes
