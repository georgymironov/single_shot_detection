from collections import defaultdict
import logging
import re

import torch

from bf.utils import torch_utils


def _to_torch_path(onnx_node):
    return '.'.join(re.findall(r'\[(.*?)\]', onnx_node.scopeName()))

def _to_tuple(val):
    if isinstance(val, list):
        return tuple(_to_tuple(x) for x in val)
    if isinstance(val, torch.Tensor):
        if val.nelement() > 1:
            return tuple(val.tolist())
        val = val.item()
    return (val,)

def _equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.eq(b).all().item()
    return a == b

def _hash_node(node):
    return hash(tuple(x.unique() for x in node.inputs()) + (node.kind(), _to_torch_path(node)) +
                tuple((attr,) + _to_tuple(node[attr]) for attr in node.attributeNames()))

def _same_node(a, b):
    return list(a.inputs()) == list(b.inputs()) and a.kind() == b.kind() and _to_torch_path(a) == _to_torch_path(b) and \
        a.attributeNames() == b.attributeNames() and all(_equal(a[attr], b[attr]) for attr in a.attributeNames())

def _is_depthwise_conv_onnx(onnx_node):
    if onnx_node.kind() != 'onnx::Conv':
        return False
    return next(onnx_node.inputs()).type().sizes()[1] == onnx_node.output().type().sizes()[1] == onnx_node['group']


class TraceInspector(object):
    _affected_in_node_types = ['onnx::Conv', 'onnx::BatchNormalization']
    _affected_out_node_types = ['onnx::Conv']

    activation_types = ['onnx::Clip', 'onnx::Elu', 'onnx::LeakyRelu', 'onnx::PRelu', 'onnx::Relu', 'onnx::Selu',
                        'onnx::Sigmoid', 'onnx::Tanh']

    def __init__(self, model):
        self.model = model
        self.trace = torch_utils.get_onnx_trace(model)  # writing to self to prevent deallocation

        graph = self.trace.graph()

        self.output_nodes = defaultdict(set)
        for node in graph.nodes():
            for inp in node.inputs():
                self.output_nodes[inp.unique()].add(node)

        nodes = {}
        self.ignore = []
        for node in graph.nodes():
            node_hash = _hash_node(node)
            if node_hash not in nodes:
                nodes[node_hash] = node
                continue

            if not _same_node(nodes[node_hash], node):
                raise KeyError('Collision occured')

            for output in node.outputs():
                for affected in self.output_nodes[output.unique()]:
                    for old_output, new_output in zip(node.outputs(), nodes[node_hash].outputs()):
                        affected.replaceInputWith(old_output, new_output)
                        logging.debug(f'DBG: REPLACED {old_output} -> {new_output}')
                self.ignore.append(output.unique())

        self.nodes = {}
        for node in graph.nodes():
            for output in node.outputs():
                self.nodes[output.unique()] = node

        self.output_nodes = defaultdict(set)
        for node in graph.nodes():
            for inp in node.inputs():
                self.output_nodes[inp.unique()].add(node)

        self.node_paths = {_to_torch_path(x): x.output().unique()
                           for x in graph.nodes()
                           if len(list(x.outputs())) == 1 and x.output().unique() not in self.ignore}

        self.affected = {k: list(self._get_affected_nodes(v, 'out')) for k, v in self.node_paths.items()}
        self.concat_groups = [list(self._get_concatenation_groups(x)) for x in graph.nodes() if x.kind() == 'onnx::Concat' and x['axis'] == 1]

    def _get_concatenation_groups(self, node):
        def _trace_multiple_inputs():
            for inp in node.inputs():
                if inp.unique() in self.nodes:
                    yield from self._get_concatenation_groups(self.nodes[inp.unique()])

        def _trace_first_input():
            inp = next(node.inputs())
            if inp.unique() in self.nodes:
                yield from self._get_concatenation_groups(self.nodes[inp.unique()])

        if node.kind() == 'onnx::Conv':
            yield _to_torch_path(node)
        elif node.kind() in ['onnx::Add', 'onnx::Concat']:
            yield from _trace_multiple_inputs()
        else:
            yield from _trace_first_input()

    @property
    def connected(self):
        return {k: [x[0] for x in v if x[1] == 'out'] for k, v in self.affected.items()}

    def get_descendent_of_type(self, path, types, stop_on=None):
        stop_on = stop_on or []
        unique = self.node_paths[path]

        for output_node in self.output_nodes[unique]:
            for output in output_node.outputs():
                yield from self._get_descendent_of_type(output.unique(), types, stop_on)

    def _get_descendent_of_type(self, unique, types, stop_on=None):
        node = self.nodes[unique]

        if node.kind() in stop_on:
            return
        if node.kind() in types:
            yield _to_torch_path(node)
        else:
            for output_node in self.output_nodes[unique]:
                for output in output_node.outputs():
                    yield from self._get_descendent_of_type(output.unique(), types, stop_on)

    def _get_affected_nodes(self, unique, type_='out', memo=None):
        if unique in self.ignore:
            return

        if memo is None:
            memo = set()

        node = self.nodes[unique]
        affected = _to_torch_path(node), type_

        if affected in memo:
            return

        def _get_affected_in_nodes():
            for output_node in self.output_nodes[unique]:
                for output in output_node.outputs():
                    yield from self._get_affected_nodes(output.unique(), 'in', memo)

        def _get_affected_out_nodes():
            for inp in node.inputs():
                if inp.unique() in self.nodes:
                    yield from self._get_affected_nodes(inp.unique(), 'out', memo)

        if _is_depthwise_conv_onnx(node):
            yield _to_torch_path(node), 'out'
            memo.add((_to_torch_path(node), 'in'))
            memo.add((_to_torch_path(node), 'out'))
            yield from _get_affected_in_nodes()
            yield from _get_affected_out_nodes()
        elif type_ == 'in':
            if node.kind() in self._affected_in_node_types and type_ == 'in':
                yield affected
                memo.add(affected)
            if node.kind() != 'onnx::Conv':
                yield from _get_affected_in_nodes()
            if node.kind() == 'onnx::Add':
                yield from _get_affected_out_nodes()
        else:
            if node.kind() in self._affected_out_node_types and type_ == 'out':
                yield affected
                memo.add(affected)
            if node.kind() == 'onnx::Conv':
                yield from _get_affected_in_nodes()
            else:
                yield from _get_affected_out_nodes()
