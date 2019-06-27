import torch
import torch.nn as nn
import torch.nn.functional as F

import bf.utils


class Predictor(nn.Module):
    def __init__(self,
                 features,
                 extras,
                 predictor,
                 heads,
                 num_classes):
        super(Predictor, self).__init__()

        self.features = features
        self.extras = extras
        self.predictor = predictor
        self.heads = heads
        self.num_classes = num_classes

    def forward(self, img):
        """
        Args:
            img: torch.tensor(:shape [Batch, Channel, Height, Width])
        Returns:
            prediction: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes * Classes])
                torch.tensor(:shape [Batch, AnchorBoxes * 4])
                torch.tensor(:shape [AnchorBoxes, 4])
        """
        scores = []
        locs = []

        with torch.jit.scope(f'{type(self.features).__name__}[features]'):
            sources, x = self.features(img)

        with torch.jit.scope('Sequential[extras]'):
            for i, layer in enumerate(self.extras):
                with torch.jit.scope(f'_item[{i}]'):
                    x = layer(x)
                sources.append(x)

        if self.predictor:
            score_sources, loc_sources = self.predictor(sources)
        else:
            score_sources = loc_sources = sources

        for i, (head, score_source, loc_source) in enumerate(zip(self.heads, score_sources, loc_sources)):
            with torch.jit.scope(f'ModuleList[heads]/ModuleDict[{i}]'):
                with torch.jit.scope(f'_item[score]'):
                    scores.append(
                        head['score'](score_source)
                            .permute((0, 2, 3, 1))
                            .contiguous()
                            .view(score_source.size(0), -1))
                with torch.jit.scope(f'_item[loc]'):
                    locs.append(
                        head['loc'](loc_source)
                            .permute((0, 2, 3, 1))
                            .contiguous()
                            .view(loc_source.size(0), -1))

        scores = torch.cat(scores, dim=1)
        locs = torch.cat(locs, dim=1)

        if bf.utils.onnx_exporter.is_exporting():
            scores = scores.view(img.size(0), -1, self.num_classes)
            scores = F.softmax(scores, dim=-1)
            scores = scores.view(img.size(0), -1)
            return scores, locs
        else:
            return scores, locs, loc_sources

class Detector(nn.Module):
    def __init__(self, *args, anchor_generators):
        super(Detector, self).__init__()
        self.predictor = Predictor(*args)
        self.priors = anchor_generators

    def generate_anchors(self, img, sources):
        anchors = []
        for loc_source, anchor_generator in zip(sources, self.priors):
            anchors.append(anchor_generator.generate(img, loc_source).view(-1))
        return torch.cat(anchors, dim=0).view(-1, 4)

    def forward(self, img):
        with torch.jit.scope('Predictor[predictor]'):
            output = self.predictor.forward(img)

        if bf.utils.onnx_exporter.is_exporting():
            return output
        else:
            scores, locs, locs_sources = output
            return scores, locs, self.generate_anchors(img, locs_sources)
