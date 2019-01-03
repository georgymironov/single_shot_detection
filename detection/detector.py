import torch
import torch.nn as nn
import torch.nn.functional as F

from bf.utils.torch_utils import get_multiple_outputs


class Detector(nn.Module):
    def __init__(self,
                 num_classes,
                 features,
                 extras,
                 heads,
                 priors,
                 source_layers):
        super(Detector, self).__init__()

        self.num_classes = num_classes
        self.features = features
        self.extras = extras
        self.heads = heads
        self.priors = priors
        self.source_layers = source_layers

        self.init()

    def forward(self, img):
        """
        Args:
            img: torch.tensor(:shape [Batch, Channel, Height, Width])
        Returns:
            prediction: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes, Classes])
                torch.tensor(:shape [Batch, AnchorBoxes, 4])
                torch.tensor(:shape [AnchorBoxes, 4])
        """
        classes = []
        locs = []
        priors = []

        sources, x = get_multiple_outputs(self.features, img, self.source_layers)

        for layer in self.extras:
            x = layer(x)
            sources.append(x)

        for head, source, prior in zip(self.heads, sources, self.priors):
            classes.append(
                head['class'](source)
                    .permute((0, 2, 3, 1))
                    .contiguous()
                    .view(source.size(0), -1, self.num_classes))

            locs.append(
                head['loc'](source)
                    .permute((0, 2, 3, 1))
                    .contiguous()
                    .view(source.size(0), -1, 4))

            priors.append(prior.forward(img, source).view(-1, 4))

        classes = torch.cat(classes, dim=1)
        locs = torch.cat(locs, dim=1)
        priors = torch.cat(priors, dim=0)

        classes = F.log_softmax(classes, dim=-1)

        return classes, locs, priors

    @staticmethod
    def init_layer(layer):
        if isinstance(layer, (nn.Conv2d)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init(self):
        self.extras.apply(self.init_layer)
        self.heads.apply(self.init_layer)
