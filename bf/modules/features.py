import torch
import torch.nn as nn

from bf.utils.torch_utils import get_multiple_outputs


class Features(nn.Module):
    def __init__(self,
                 base,
                 out_layers,
                 last_feature_layer=None):
        super(Features, self).__init__()

        assert isinstance(base.features, nn.Sequential)

        feature_layers = list(base.features.children())
        if last_feature_layer is not None:
            feature_layers = feature_layers[:(last_feature_layer + 1)]

        self.base = nn.Sequential(*feature_layers)
        self.out_layers = out_layers

    def forward(self, x):
        with torch.jit.scope('Sequential[base]'):
            sources, x = get_multiple_outputs(self.base, x, self.out_layers)
        return sources, x

    def get_out_channels(self):
        dummy = torch.ones((1, 3, 300, 300), dtype=torch.float)
        sources, _ = get_multiple_outputs(self.base, dummy, self.out_layers)
        return [x.size(1) for x in sources]
