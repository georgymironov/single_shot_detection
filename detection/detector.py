import torch
import torch.nn as nn
import torch.nn.functional as F

import bf.utils


class Predictor(nn.Module):
    def __init__(self,
                 features,
                 extras,
                 predictor,
                 heads):
        super(Predictor, self).__init__()

        self.features = features
        self.extras = extras
        self.predictor_conv = predictor[0]
        self.predictor_activation = predictor[1]
        self.predictor_norm = predictor[2]
        self.heads = heads

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

        score_sources = loc_sources = sources

        for score_conv, loc_conv, score_norm, loc_norm in zip(self.predictor_conv['score'],
                                                              self.predictor_conv['loc'],
                                                              self.predictor_norm['score'],
                                                              self.predictor_norm['loc']):
            score_sources = map(score_conv, score_sources)
            loc_sources = map(loc_conv, loc_sources)

            score_sources = map(self.predictor_activation, score_sources)
            loc_sources = map(self.predictor_activation, loc_sources)

            score_sources = [norm(x) for norm, x in zip(score_norm, score_sources)]
            loc_sources = [norm(x) for norm, x in zip(loc_norm, loc_sources)]

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
            return scores, locs
        else:
            return scores, locs, loc_sources

class _ScriptDetector(torch.jit.ScriptModule):
    def __init__(self, traced_predictor, anchors):
        super(_ScriptDetector, self).__init__()
        self.anchors = nn.Parameter(anchors)
        self.traced_predictor = traced_predictor

    @torch.jit.script_method
    def forward(self, x):
        return (*self.traced_predictor(x), self.anchors)

class Detector(nn.Module):
    def __init__(self, *args, num_classes, anchor_generators):
        super(Detector, self).__init__()
        self.predictor = Predictor(*args)
        self.num_classes = num_classes
        self.priors = anchor_generators

    def jit(self, x):
        scores, locs, anchors = self.forward(x)

        with bf.utils.onnx_exporter.for_export():
            traced_predictor = torch.jit.trace(self.predictor, x)

        return _ScriptDetector(traced_predictor, anchors)

    def generate_anchors(self, img, sources):
        anchors = []
        for loc_source, anchor_generator in zip(sources, self.priors):
            anchors.append(anchor_generator.generate(img, loc_source).view(-1))
        return torch.cat(anchors, dim=0).view(-1, 4)

    def forward(self, img):
        with torch.jit.scope('Predictor[predictor]'):
            output = self.predictor.forward(img)

        if bf.utils.onnx_exporter.is_exporting():
            scores, locs = output
            scores = scores.view(img.size(0), -1, self.num_classes)
            scores = F.softmax(scores, dim=-1)
            scores = scores.view(img.size(0), -1)
            return scores, locs
        else:
            scores, locs, locs_sources = output
            return scores, locs, self.generate_anchors(img, locs_sources)
