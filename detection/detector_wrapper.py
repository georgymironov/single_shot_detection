from bf.preprocessing.transforms import Resize


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

    def predict_single(self, input_):
        ratio_w = input_.shape[1] / self.input_size[0]
        ratio_h = input_.shape[0] / self.input_size[1]

        with self.preprocess.context('no_target'):
            input_ = self.preprocess(input_)
        if input_.dim() == 3:
            input_ = input_.unsqueeze(0)

        *prediction, priors = self.model(input_.to(self.device))
        prediction = [x.detach() for x in prediction]
        result = self.postprocessor.postprocess(prediction, priors)[0]

        result[..., [0, 2]] *= ratio_w
        result[..., [1, 3]] *= ratio_h

        return result
