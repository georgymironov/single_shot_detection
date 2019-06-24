import torch


def export(model, input_size, path):
    model.eval()
    device = next(model.parameters()).device
    data = torch.rand((1, 3, input_size[1], input_size[0]), dtype=torch.float32, device=device)

    if hasattr(model, 'jit'):
        model.jit(data).save(path)
    else:
        torch.jit.trace(model, data).save(path)
