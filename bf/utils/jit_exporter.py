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

def export(model, input_size, path):
    model.eval()
    device = next(model.parameters()).device
    data = torch.rand((1, 3, input_size[1], input_size[0]), dtype=torch.float32, device=device)
    with for_export():
        torch.jit.trace(model, data).save(path)
