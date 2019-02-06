import bf.preprocessing


class DetectorWrapper(object):
    def __init__(self, detector, postprocessor, preprocess=None, resize=None):
        self.device = next(detector.parameters()).device
        self.model = detector
        self.postprocessor = postprocessor
        self.preprocess = preprocess
        self.resize = resize

    def predict_single(self, input_):
        ratio_w = input_.shape[1] / self.resize.size[0]
        ratio_h = input_.shape[0] / self.resize.size[1]

        with bf.preprocessing.set_transform_type('no_target'):
            if self.resize is not None:
                input_ = self.resize(input_)
            if self.preprocess is not None:
                input_ = self.preprocess(input_)
        if input_.dim() == 3:
            input_ = input_.unsqueeze(0)

        *prediction, priors = self.model(input_.to(self.device))
        prediction = [x.detach() for x in prediction]
        result = self.postprocessor.postprocess(prediction, priors)[0]

        result[..., [0, 2]] *= ratio_w
        result[..., [1, 3]] *= ratio_h

        return result
