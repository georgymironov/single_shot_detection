import torch

from bf.utils import onnx_exporter
from bf.preprocessing.transforms import Resize


class _ScriptDetector(torch.jit.ScriptModule):
    def __init__(self, traced_predictor, anchors, box_coder):
        super(_ScriptDetector, self).__init__()
        self.anchors = torch.nn.Parameter(anchors)
        self.traced_predictor = traced_predictor
        self.box_coder = box_coder

    @torch.jit.script_method
    def forward(self, x):
        scores, locs = self.traced_predictor(x)

        scores = scores.view(scores.size(0), self.anchors.size(0), -1)

        locs = locs.view(locs.size(0), self.anchors.size(0), 4)
        locs = self.box_coder.decode_box(locs, self.anchors, inplace=torch.tensor(0))

        return scores, locs

class DetectorWrapper(object):
    def __init__(self, detector, preprocess, postprocessor):
        self.device = next(detector.parameters()).device
        self.model = detector
        self.preprocess = preprocess
        self.postprocessor = postprocessor

        for transform in preprocess.transforms:
            if isinstance(transform, Resize):
                self.input_size = transform.size
                break

    def eval(self):
        self.model.eval()

    def jit(self, x):
        _, _, anchors = self.model.forward(x)

        with onnx_exporter.for_export():
            traced_predictor = torch.jit.trace(self.model.predictor, x)

        return _ScriptDetector(traced_predictor, anchors, self.postprocessor.box_coder)

    def predict_single(self, img):
        assert img.dim() == 3
        ratio_w = img.shape[1] / self.input_size[0]
        ratio_h = img.shape[0] / self.input_size[1]

        with self.preprocess.context('no_target'):
            img = self.preprocess(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            *prediction, priors = self.model(img.to(self.device))
            result = self.postprocessor.postprocess(prediction, priors)[0]

        result[..., [0, 2]] *= ratio_w
        result[..., [1, 3]] *= ratio_h

        return result
