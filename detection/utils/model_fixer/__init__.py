from .rules import RULE_REGISTRY


def fix_weights(weights):
    for fixes in RULE_REGISTRY:
        weights = fixes(weights)
    return weights
