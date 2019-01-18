import torch
import torch.nn as nn
import torch.nn.functional as F

import bf.preprocessing
import bf.utils
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
                torch.tensor(:shape [Batch, AnchorBoxes * Classes])
                torch.tensor(:shape [Batch, AnchorBoxes * 4])
                torch.tensor(:shape [AnchorBoxes, 4])
        """
        scores = []
        locs = []
        priors = []

        with torch.jit.scope('Sequential[features]'):
            sources, x = get_multiple_outputs(self.features, img, self.source_layers)

        with torch.jit.scope('Sequential[extras]'):
            for i, layer in enumerate(self.extras):
                with torch.jit.scope(f'_item[{i}]'):
                    x = layer(x)
                sources.append(x)

        for i, (head, source, prior) in enumerate(zip(self.heads, sources, self.priors)):
            with torch.jit.scope(f'ModuleList[heads]/ModuleDict[{i}]'):
                with torch.jit.scope(f'_item[class]'):
                    scores.append(
                        head['class'](source)
                            .permute((0, 2, 3, 1))
                            .contiguous()
                            .view(source.size(0), -1))
                with torch.jit.scope(f'_item[loc]'):
                    locs.append(
                        head['loc'](source)
                            .permute((0, 2, 3, 1))
                            .contiguous()
                            .view(source.size(0), -1))

            if not bf.utils.onnx_exporter.is_exporting():
                priors.append(prior.generate(img, source).view(-1))

        scores = torch.cat(scores, dim=1)
        locs = torch.cat(locs, dim=1)

        if not bf.utils.onnx_exporter.is_exporting():
            priors = torch.cat(priors, dim=0).view(-1, 4)

        scores = scores.view(img.size(0), -1, self.num_classes)
        scores = F.softmax(scores, dim=-1) if bf.utils.onnx_exporter.is_exporting() else F.log_softmax(scores, dim=-1)
        scores = scores.view(img.size(0), -1)

        if not bf.utils.onnx_exporter.is_exporting():
            return scores, locs, priors
        else:
            return scores, locs

    @staticmethod
    def init_layer(layer):
        if isinstance(layer, (nn.Conv2d)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init(self):
        self.extras.apply(self.init_layer)
        self.heads.apply(self.init_layer)


class DetectorWrapper(object):
    def __init__(self, detector, postprocessor, preprocess=None):
        self.device = next(detector.parameters()).device
        self.model = detector
        self.postprocessor = postprocessor
        self.preprocess = preprocess

    def predict_single(self, input_):
        if self.preprocess is not None:
            with bf.preprocessing.set_transform_type('no_target'):
                input_ = self.preprocess(input_)
        if input_.dim() == 3:
            input_ = input_.unsqueeze(0)
        *prediction, priors = self.model(input_.to(self.device))
        prediction = [x.detach() for x in prediction]
        return self.postprocessor.postprocess(prediction, priors)[0]
