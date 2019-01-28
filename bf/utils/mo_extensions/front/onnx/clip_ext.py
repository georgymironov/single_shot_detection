from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.clamp import Clamp


class AddFrontExtractor(FrontExtractorOp):
    op = 'Clip'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = {
            'max': onnx_attr(node, 'max', 'f', default=None),
            'min': onnx_attr(node, 'min', 'f', default=None),
        }
        Clamp.update_node_stat(node, attrs)
        return __class__.enabled
