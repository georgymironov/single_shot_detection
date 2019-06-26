import torch


def export(model, input_size, path):
    model.eval()
    data = torch.rand((1, 3, input_size[1], input_size[0]), dtype=torch.float32)

    if hasattr(model, 'jit'):
        model.jit(data).save(path)
    else:
        torch.jit.trace(model, data).save(path)
