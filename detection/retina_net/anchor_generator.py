import functools
import math

import torch

from bf.utils.misc_utils import filter_kwargs


@filter_kwargs
def get_priors(aspect_ratios,
               min_level,
               max_level,
               scale,
               scales_per_level):
    return [AnchorGenerator(aspect_ratios, level, scale, scales_per_level) for level in range(min_level, max_level + 1)]

class AnchorGenerator(object):
    def __init__(self,
                 aspect_ratios,
                 level,
                 scale,
                 scales_per_level=1):
        self.num_boxes = len(aspect_ratios) * scales_per_level
        self.device = 'cpu'

        sizes = [scale * (2 ** (level + x / scales_per_level)) for x in range(scales_per_level)]
        self.hws = torch.empty((self.num_boxes, 2), dtype=torch.float32, device=self.device)

        for j, size in enumerate(sizes):
            for i, ar in enumerate(aspect_ratios):
                self.hws[j * len(aspect_ratios) + i][0] = size * math.sqrt(ar)
                self.hws[j * len(aspect_ratios) + i][1] = size / math.sqrt(ar)

    @functools.lru_cache()
    def _generate_anchors(self, img_size, feature_map_size):
        img_w, img_h = img_size
        layer_w, layer_h = feature_map_size

        boxes = torch.empty((layer_h, layer_w, self.num_boxes, 4), dtype=torch.float32, device=self.device)

        step_w = img_w / layer_w
        step_h = img_h / layer_h

        xs = torch.linspace(0.5 * step_w, (0.5 + layer_w - 1) * step_w, layer_w, device=self.device)
        ys = torch.linspace(0.5 * step_h, (0.5 + layer_h - 1) * step_h, layer_h, device=self.device)
        y_grid, x_grid = torch.meshgrid(ys, xs)

        boxes[..., 0] = x_grid.unsqueeze_(-1)
        boxes[..., 1] = y_grid.unsqueeze_(-1)
        boxes[..., 2] = self.hws[..., 0]
        boxes[..., 3] = self.hws[..., 1]

        return boxes

    def generate(self, img, feature_map):
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
