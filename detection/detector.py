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
                 source_layers):
        super(Predictor, self).__init__()

        self.features = features
        self.extras = extras
        self.predictor_conv = predictor[0]
        self.predictor_activation = predictor[1]
        self.predictor_norm = predictor[2]
        self.heads = heads
        self.source_layers = source_layers

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

        # backward compatibility
        # ToDo: remove
        if isinstance(self.features, nn.Sequential):
            from bf.utils.torch_utils import get_multiple_outputs
            with torch.jit.scope('Sequential[features]'):
                sources, x = get_multiple_outputs(self.features, img, self.source_layers)
        else:
            with torch.jit.scope(f'{type(self.features).__name__}[features]'):
                sources, x = self.features(img)

        with torch.jit.scope('Sequential[extras]'):
            for i, layer in enumerate(self.extras):
                with torch.jit.scope(f'_item[{i}]'):
                    x = layer(x)
                sources.append(x)

        score_sources = loc_sources = sources

        # backward compatibility
        # ToDo: remove
        if hasattr(self, 'predictor_conv'):
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

        return scores, locs, loc_sources

class Detector(nn.Module):
    def __init__(self, *args, num_classes, anchor_generators):
        super(Detector, self).__init__()
        self.predictor = Predictor(*args)
        self.num_classes = num_classes
        self.priors = anchor_generators

    def forward(self, img):
        with torch.jit.scope('Predictor[predictor]'):
            scores, locs, locs_sources = self.predictor.forward(img)
        priors = []

        if bf.utils.jit_exporter.is_exporting():
            return scores, locs, tuple(locs_sources)
        if bf.utils.onnx_exporter.is_exporting():
            scores = scores.view(img.size(0), -1, self.num_classes)
            scores = F.softmax(scores, dim=-1)
            scores = scores.view(img.size(0), -1)
            return scores, locs
        else:
            for loc_source, anchor_generator in zip(locs_sources, self.priors):
                priors.append(anchor_generator.generate(img, loc_source).view(-1))
            priors = torch.cat(priors, dim=0).view(-1, 4)
            return scores, locs, priors
