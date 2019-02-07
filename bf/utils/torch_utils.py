import torch
import torch.nn as nn

from . import onnx_exporter


def get_multiple_outputs(model, input_, output_layers):
    assert isinstance(model, nn.Sequential)

    x = input_
    output_layer_idx = 0
    outputs = []

    for i, layer in enumerate(model):
        with torch.jit.scope(f'_item[{i}]'):
            x = layer(x)
        output_layer = output_layers[output_layer_idx]

        if isinstance(output_layer, int):
            if i == output_layer:
                outputs.append(x)
                output_layer_idx += 1
        elif isinstance(output_layer, list):
            if i == output_layer[0]:
                with torch.jit.scope(f'_item[{i+1}]'):
                    y = x
                    for name, child in model[i + 1].named_children():
                        with torch.jit.scope(f'{type(child).__name__}[{name}]'):
                            y = child(y)
                        if name == output_layer[1]:
                            break
                    else:
                        raise ValueError(f'Wrong layer {output_layer}')
                outputs.append(y)
                output_layer_idx += 1

    return outputs, x

def get_leaf_modules(module, memo=None, prefix=''):
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        if not module._modules:
            yield prefix, module
        else:
            for name, submodule in module._modules.items():
                if submodule is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in get_leaf_modules(submodule, memo, submodule_prefix):
                    yield m

def get_onnx_trace(model, input_=None):
    with onnx_exporter.for_export():
        if input_ is None:
            input_ = torch.ones((2, 3, 224, 224), dtype=torch.float)
        device = next(model.parameters()).device
        input_ = input_.to(device)
        trace, _ = torch.jit.get_trace_graph(model, input_)
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        return trace
