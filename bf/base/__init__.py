import functools

from torchvision.models import *

from .mobilenet_v2 import MobileNetV2

def _wrapped_partial(func, *args, **kwargs):
    return functools.update_wrapper(functools.partial(func, *args, **kwargs), func)

mobilenet_v2_10 = _wrapped_partial(MobileNetV2, depth_multiplier=1.)
mobilenet_v2_075 = _wrapped_partial(MobileNetV2, depth_multiplier=.75)
mobilenet_v2_050 = _wrapped_partial(MobileNetV2, depth_multiplier=.5)
mobilenet_v2_035 = _wrapped_partial(MobileNetV2, depth_multiplier=.35)

del mobilenet_v2