import bf.preprocessing


class DetectorWrapper(object):
    def __init__(self, detector, postprocessor, preprocess=None, resize=None):
        self.device = next(detector.parameters()).device
        self.model = detector
        self.postprocessor = postprocessor
        self.preprocess = preprocess
        self.resize = resize

    def predict_single(self, input_):
        with bf.preprocessing.set_transform_type('no_target'):
            if self.resize is not None:
                input_ = self.resize(input_)
            if self.preprocess is not None:
                input_ = self.preprocess(input_)
        if input_.dim() == 3:
            input_ = input_.unsqueeze(0)
        *prediction, priors = self.model(input_.to(self.device))
        prediction = [x.detach() for x in prediction]
        return self.postprocessor.postprocess(prediction, priors)[0]
