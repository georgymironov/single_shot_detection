import functools

from torchvision.models import resnet18 as torchvision_resnet18
from torchvision.models import resnet34 as torchvision_resnet34
from torchvision.models import resnet50 as torchvision_resnet50
from torchvision.models import resnet101 as torchvision_resnet101
from torchvision.models import resnet152 as torchvision_resnet152

from torchvision.models import resnext50_32x4d as torchvision_resnext50_32x4d
from torchvision.models import resnext101_32x8d as torchvision_resnext101_32x8d

from torchvision.models import shufflenet_v2_x0_5 as torchvision_shufflenet_v2_x0_5
from torchvision.models import shufflenet_v2_x1_0 as torchvision_shufflenet_v2_x1_0
from torchvision.models import shufflenet_v2_x1_5 as torchvision_shufflenet_v2_x1_5
from torchvision.models import shufflenet_v2_x2_0 as torchvision_shufflenet_v2_x2_0

from torchvision.models import vgg11 as torchvision_vgg11
from torchvision.models import vgg11_bn as torchvision_vgg11_bn
from torchvision.models import vgg13 as torchvision_vgg13
from torchvision.models import vgg13_bn as torchvision_vgg13_bn
from torchvision.models import vgg16 as torchvision_vgg16
from torchvision.models import vgg16_bn as torchvision_vgg16_bn
from torchvision.models import vgg19 as torchvision_vgg19
from torchvision.models import vgg19_bn as torchvision_vgg19_bn

try:
    from pretrainedmodels.models import resnext101_32x4d as pretrainedmodels_resnext101_32x4d
    from pretrainedmodels.models import resnext101_64x4d as pretrainedmodels_resnext101_64x4d

    from pretrainedmodels.models import se_resnet50 as pretrainedmodels_se_resnet50
    from pretrainedmodels.models import se_resnet101 as pretrainedmodels_se_resnet101
    from pretrainedmodels.models import se_resnet152 as pretrainedmodels_se_resnet152
    from pretrainedmodels.models import se_resnext50_32x4d as pretrainedmodels_se_resnext50_32x4d
    from pretrainedmodels.models import se_resnext101_32x4d as pretrainedmodels_se_resnext101_32x4d
except Exception:
    pass

from .mobilenet import MobileNet
from .mobilenet_v2 import MobileNetV2

def _wrapped_partial(func, *args, **kwargs):
    return functools.update_wrapper(functools.partial(func, *args, **kwargs), func)

mobilenet_10 = _wrapped_partial(MobileNet, depth_multiplier=1.)
mobilenet_075 = _wrapped_partial(MobileNet, depth_multiplier=.75)
mobilenet_050 = _wrapped_partial(MobileNet, depth_multiplier=.5)
mobilenet_025 = _wrapped_partial(MobileNet, depth_multiplier=.25)

mobilenet_v2_10 = _wrapped_partial(MobileNetV2, depth_multiplier=1.)
mobilenet_v2_075 = _wrapped_partial(MobileNetV2, depth_multiplier=.75)
mobilenet_v2_050 = _wrapped_partial(MobileNetV2, depth_multiplier=.5)
mobilenet_v2_035 = _wrapped_partial(MobileNetV2, depth_multiplier=.35)

del mobilenet
del mobilenet_v2
