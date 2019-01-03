def get_multiple_outputs(model, input_, output_layers):
    x = input_
    output_layer_idx = 0
    outputs = []

    for i, layer in enumerate(model):
        x = layer(x)
        output_layer = output_layers[output_layer_idx]

        if isinstance(output_layer, int):
            if i == output_layer:
                outputs.append(x)
                output_layer_idx += 1
        elif isinstance(output_layer, list):
            if i == output_layer[0]:
                y = x
                for name, child in model[i + 1].named_children():
                    y = child(y)
                    if name == output_layer[1]:
                        break
                else:
                    raise ValueError(f'Wrong layer {output_layer}')
                outputs.append(y)
                output_layer_idx += 1

    return outputs, x
