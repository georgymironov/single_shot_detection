from collections import OrderedDict


RULE_REGISTRY = []

def register(func):
    RULE_REGISTRY.append(func)
    return func

@register
def rename_class_to_score(weights):
    return OrderedDict({k.replace('.class.', '.score.'): v for k, v in weights.items()})

@register
def convert_from_distributed(weights):
    return OrderedDict({k.replace('predictor.module.', 'predictor.'): v for k, v in weights.items()})
