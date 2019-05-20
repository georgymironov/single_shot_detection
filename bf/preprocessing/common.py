import random

from bf.preprocessing import no_target, functional

from . import transforms as _transforms


class Transform(object):
    def __call__(self, sample):
        return self.apply(sample)

    def apply(self, sample):
        raise NotImplementedError

class DynamicTransform(object):
    def __init__(self, transform_type='no_target'):
        self.set_target_type(transform_type)

    def set_target_type(self, transform_type):
        if transform_type == 'box':
            self.target_functional = functional.box_fn
        elif transform_type == 'no_target':
            self.target_functional = no_target
        else:
            raise ValueError(f'Unknown transform_type: {transform_type}')
        return self

    def __call__(self, sample):
        dummy_target = False

        if not isinstance(sample, tuple):
            sample = sample, None
            dummy_target = True

        result = self.apply(sample)
        if dummy_target:
            result = result[0]

        return result

    @property
    def _no_target(self):
        return self.target_functional is no_target

    def apply(self, sample):
        raise NotImplementedError

class RandomDynamicTransform(DynamicTransform):
    def __init__(self, p=.5):
        super(RandomDynamicTransform, self).__init__()
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return super(RandomDynamicTransform, self).__call__(sample)
        return sample

class RandomTransform(Transform):
    def __init__(self, p=.5):
        super(RandomTransform, self).__init__()
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return super(RandomTransform, self).__call__(sample)
        return sample

class _ContainerContext(object):
    def __init__(self, that, transform_type):
        self.that = that
        self.transform_type = transform_type

    def __enter__(self):
        self.old = self.that.transform_type
        self.that.set_target_type(self.transform_type)

    def __exit__(self, *args):
        self.that.set_target_type(self.old)

class TransformContainer(object):
    def __init__(self, transforms, transform_type='no_target'):
        self.transforms = [getattr(_transforms, x['name'])(**x.get('args', {})) for x in transforms]
        self.transform_type = transform_type
        self.set_target_type(transform_type)

    def set_target_type(self, transform_type):
        self.transform_type = transform_type
        for transform in self.transforms:
            if isinstance(transform, (DynamicTransform, TransformContainer)):
                transform.set_target_type(transform_type)
        return self

    def context(self, transform_type):
        return _ContainerContext(self, transform_type)
