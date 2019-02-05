from collections import defaultdict
import logging
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


def _node_str(node):
    assert(len(node.output) == 1)
    return f'#{node.output[0]} {node.op_type}'

def _hash_node(node):
    return hash(tuple(node.input) + (node.op_type,) + tuple(x.SerializeToString() for x in node.attribute))

def _same_node(node_a, node_b):
    return node_a.input == node_b.input and node_a.op_type == node_b.op_type and node_a.attribute == node_b.attribute

def _replace_input(node, old, new):
    index = list(node.input).index(old)
    node.input.insert(index, new)
    node.input.remove(old)
    logging.debug(f'{_node_str(node)} replace input: {old} -> {new}')

def _merge_nodes(graph):
    output_nodes = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            output_nodes[inp].append(node)

    nodes = {}
    to_remove = []
    for node in graph.node:
        _hash = _hash_node(node)

        if _hash not in nodes:
            nodes[_hash] = node
            continue

        if not _same_node(nodes[_hash], node):
            raise KeyError('Collision occured')

        old_output = node.output
        assert(len(old_output) == 1)
        old_output = old_output[0]

        new_output = nodes[_hash].output
        assert(len(new_output) == 1)
        new_output = new_output[0]

        to_remove.append(node)
        logging.debug(f'{_node_str(node)} removed')

        affected = output_nodes[old_output]
        for aff in affected:
            _replace_input(aff, old_output, new_output)

        logging.debug('-' * 25)

    for node in to_remove:
        graph.node.remove(node)

def export(model, input_size, filename):
    model.eval()
    device = next(model.parameters()).device
    data = torch.rand((1, 3, input_size[1], input_size[0]), dtype=torch.float32, device=device)

    logging.info('===> Exporting to ONNX...')
    _, tmp = tempfile.mkstemp()

    with for_export():
        torch.onnx.export(model, data, tmp)

    model = onnx.load(tmp)
    graph = model.graph

    _merge_nodes(graph)

    onnx.save(model, filename)
