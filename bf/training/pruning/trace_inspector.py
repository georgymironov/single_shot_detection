from collections import defaultdict
import re

from bf.utils import torch_utils


def _to_torch_path(onnx_node):
    return '.'.join(re.findall(r'\[(.*?)\]', onnx_node.scopeName()))

def _is_depthwise_conv_onnx(onnx_node):
    if onnx_node.kind() != 'onnx::Conv':
        return False
    return next(onnx_node.inputs()).type().sizes()[1] == onnx_node.output().type().sizes()[1] == onnx_node['group']


class TraceInspector(object):
    _affected_in_node_types = ['onnx::Conv', 'onnx::BatchNormalization']
    _affected_out_node_types = ['onnx::Conv']

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
            path = _to_torch_path(node)
            if path not in nodes:
                nodes[path] = node
                continue

            for output in node.outputs():
                for affected in self.output_nodes[output.unique()]:
                    for old_output, new_output in zip(node.outputs(), nodes[path].outputs()):
                        affected.replaceInputWith(old_output, new_output)
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
                           if x.kind() == 'onnx::Conv' and x.output().unique() not in self.ignore}

        self.affected = {k: list(self.get_affected_nodes(v, 'out')) for k, v in self.node_paths.items()}

    @property
    def connected(self):
        return {k: [x[0] for x in v if x[1] == 'out'] for k, v in self.affected.items()}

    def get_affected_nodes(self, unique, type_='out', memo=None):
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
                    yield from self.get_affected_nodes(output.unique(), 'in', memo)

        def _get_affected_out_nodes():
            for inp in node.inputs():
                if inp.unique() in self.nodes:
                    yield from self.get_affected_nodes(inp.unique(), 'out', memo)

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
