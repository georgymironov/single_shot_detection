import functools
import logging
import math

import torch

from bf.utils.misc_utils import filter_kwargs
from ._anchor_generator import _AnchorGenerator


@filter_kwargs
def build_anchor_generators(num_scales=6,
                            sizes=None,
                            min_scale=None,
                            max_scale=None,
                            aspect_ratios=[[1.0, 2.0]] + [[1.0, 2.0, 3.0]] * 3 + [[1.0, 2.0]] * 2,
                            steps=None,
                            offsets=[0.5, 0.5],
                            num_branches=None):
    assert sizes is not None or (min_scale is not None and max_scale is not None)

    if steps is None:
        steps = [None] * num_scales
    else:
        assert len(steps) == num_scales

    if num_branches is None:
        num_branches = [1] * num_scales
    else:
        assert len(num_branches) == num_scales

    if min_scale is not None and max_scale is not None:
        scales = torch.linspace(min_scale, max_scale, num_scales + 1)
        logging.info(f'Detector (Scales: {scales[:-1]})')
    else:
        scales = None

    assert len(aspect_ratios) == num_scales

    anchor_generators = []
    for i, (ratios, step, num_branches) in enumerate(zip(aspect_ratios, steps, num_branches)):
        if scales is not None:
            kwargs = {
                'min_scale': scales[i],
                'max_scale': scales[i + 1]
            }
        else:
            kwargs = {
                'min_size': sizes[i],
                'max_size': sizes[i + 1]
            }
        anchor_generators.append(SsdAnchorGenerator(ratios, step=step, num_branches=num_branches, **kwargs))
    return anchor_generators

class SsdAnchorGenerator(_AnchorGenerator):
    def __init__(self,
                 aspect_ratios,
                 min_scale=None,
                 max_scale=None,
                 min_size=None,
                 max_size=None,
                 step=None,
                 offset=[.5, .5],
                 num_branches=1,
                 flip=True,
                 clip=False):
        super(SsdAnchorGenerator, self).__init__()

        if max_scale is not None and min_scale is None:
            raise ValueError('"max_scale" should be provided along with "min_scale"')

        if max_size is not None and min_size is None:
            raise ValueError('"max_size" should be provided along with "min_size"')

        if min_scale is not None and min_size is not None:
            raise ValueError('Either "min_scale" or "min_size" should be provided')

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_size = min_size
        self.max_size = max_size
        self.num_branches = num_branches
        self.clip = clip
        self.offset = offset
        self.step = step

        self.aspect_ratios = []
        for ar in aspect_ratios:
            assert ar >= 1.0 or not flip
            self.aspect_ratios.append(ar)
            if ar > 1.0 and flip:
                self.aspect_ratios.append(1.0 / ar)

        self.num_ratios = len(self.aspect_ratios)

        if max_scale or max_size:
            self.num_ratios += 1

        self.num_boxes = self.num_ratios * num_branches

        if self.min_size is not None and self.max_size is not None:
            self.sizes = torch.linspace(self.min_size, self.max_size, self.num_branches + 1).unsqueeze(1).expand(-1, 2)
        else:
            self.scales = torch.linspace(self.min_scale, self.max_scale, self.num_branches + 1).unsqueeze(1)

    @functools.lru_cache()
    def _generate_anchors(self, img_size, feature_map_size, device='cpu'):
        img_w, img_h = img_size
        layer_w, layer_h = feature_map_size

        boxes = torch.empty((layer_h, layer_w, self.num_boxes, 4), dtype=torch.float32, device=device)

        if self.step is not None:
            step_w = self.step
            step_h = self.step
        else:
            step_w = img_w / layer_w
            step_h = img_h / layer_h

        hws = torch.empty((self.num_boxes, 2), dtype=torch.float32, device=device)

        if self.min_size is not None and self.max_size is not None:
            sizes = self.sizes
        else:
            sizes = torch.cat([self.scales * img_w, self.scales * img_h], dim=1)

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
