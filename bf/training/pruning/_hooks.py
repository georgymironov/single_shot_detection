import torch


def _save_output_hook(module, input, output):
    module.output = output.detach()

class _moving_average_hook(object):
    def __init__(self, pruned_module, momentum=0.9):
        self.momentum = momentum
        self.pruned_module = pruned_module

    def __call__(self, gate_module, *args):
        with torch.no_grad():
            value = self.get_value(gate_module, *args)
            if 'pruning_criterion' not in self.pruned_module._buffers:
                self.pruned_module.register_buffer('pruning_criterion', value)
            else:
                self.pruned_module.pruning_criterion *= self.momentum
                self.pruned_module.pruning_criterion += (1.0 - self.momentum) * value

class _mean_activation_hook(_moving_average_hook):
    def get_value(self, module, input, output):
        return output.mean(dim=(0, 2, 3))

class _taylor_expansion_hook(_moving_average_hook):
    def get_value(self, module, grad_input, grad_output):
        value = (grad_output[0] * module.output).abs().mean(dim=(0, 2, 3))
        value /= (torch.norm(value) + 1e-8)
        return value
