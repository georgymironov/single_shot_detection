import tempfile

import onnx
import torch


_exporting = False

def is_exporting():
    return _exporting

class for_export(object):
    def __init__(self):
        global _exporting
        self.prev = _exporting
        _exporting = True

    def __enter__(self):
        pass

    def __exit__(self, *args):
        global _exporting
        _exporting = self.prev


def _same_tensor(node_a, node_b):
    return node_a.input == node_b.input and node_a.op_type == node_b.op_type and node_a.attribute == node_b.attribute

def export(model, data, filename):
    model.eval()
    _, tmp = tempfile.mkstemp()
    with for_export():
        torch.onnx.export(model, data, tmp)
    model = onnx.load(tmp)
    # ToDo: add some processing
    onnx.save(model, filename)
