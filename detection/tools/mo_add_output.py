import logging
import operator
import os
import sys

import bs4
import lxml.etree as etree
import networkx as nx
import numpy as np

from bf.training import helpers


def EDGE(*args):
    return bs4.BeautifulSoup('<edge from-layer="{}" from-port="{}" to-layer="{}" to-port="{}"/>'.format(*args), 'lxml-xml')

def PRIOR_BOX(**kwargs):
    return bs4.BeautifulSoup('''
<layer id="{id}" name="priorbox_{i}" precision="FP32" type="PriorBoxV2">
    <data aspect_ratio="{aspect_ratio}" clip="0" flip="1" max_size="{max_size}" min_size="{min_size}" offset="0.5" variance="{variance}"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>{num_channel}</dim>
            <dim>{filter_height}</dim>
            <dim>{filter_width}</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>3</dim>
            <dim>{input_height}</dim>
            <dim>{input_width}</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>2</dim>
            <dim>{size_boxes}</dim>
        </port>
    </output>
</layer>'''.format(**kwargs), 'lxml-xml')

def CONCAT(**kwargs):
    return bs4.BeautifulSoup('''
<layer id="{id}" name="concat_priorbox" precision="FP32" type="Concat">
    <data axis="2"/>
    <input>
        {input_ports}
    </input>
    <output>
        {output_ports}
    </output>
</layer>'''.format(**kwargs), 'lxml-xml')

def PORT(**kwargs):
    return '<port id="{id}">{dims}</port>'.format(**kwargs)

def DIMS(*args):
    DIM = '<dim>{size}</dim>'
    return ''.join(DIM.format(size=x) for x in args)

def DETECTION_OUTPUT(**kwargs):
    return bs4.BeautifulSoup('''
<layer id="{id}" name="detection_out" precision="FP32" type="DetectionOutput">
    <data background_label_id="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE"
          confidence_threshold="{score_threshold}" eta="1.0" input_height="1" input_width="1"
          keep_top_k="{max_total}" nms_threshold="{overlap_threshold}" normalized="1"
          num_classes="{num_classes}" share_location="1" top_k="{max_per_class}"
          variance_encoded_in_target="0" visualize="False"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>{size_locs}</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>{size_classes}</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>2</dim>
            <dim>{size_boxes}</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>1</dim>
            <dim>1</dim>
            <dim>{max_total}</dim>
            <dim>7</dim>
        </port>
    </output>
</layer>'''.format(**kwargs), 'lxml-xml')


def add_output(model_file, config):
    logging.info(f'===> {os.path.relpath(model_file, os.getcwd())}: adding output for detection model')

    with open(model_file, 'r') as f:
        model = bs4.BeautifulSoup(f, 'lxml-xml')

    g = nx.DiGraph()

    layers_root = model.net.layers
    layers = [x for x in layers_root.children if x.name == 'layer']
    layers_by_id = {x['id']: x for x in layers}

    g.add_nodes_from(x['id'] for x in layers)

    edges_root = model.net.edges
    edges = [x for x in edges_root.children if x.name == 'edge']

    g.add_edges_from((x['from-layer'], x['to-layer']) for x in edges)

    inp = [x[0] for x in g.pred.items() if len(x[1]) == 0]
    assert len(inp) == 1
    inp = inp[0]

    outputs = [x[0] for x in g.succ.items() if len(x[1]) == 0]
    assert len(outputs) == 2

    softmax = [x['id'] for x in layers if x['type'] == 'SoftMax']
    assert len(softmax) == 1
    softmax = softmax[0]

    if nx.algorithms.has_path(g, softmax, outputs[0]):
        score_output, locs_output = outputs
    else:
        locs_output, score_output = outputs

    concats = [x['id'] for x in layers if x['type'] == 'Concat']
    assert len(concats) == 2

    if nx.algorithms.has_path(g, concats[0], score_output):
        score_concat, locs_concat = concats
    else:
        locs_concat, score_concat = concats

    score_heads = [x[0] for x in g.in_edges(score_concat)]
    head_distances = [nx.algorithms.shortest_path_length(g, inp, x) for x in score_heads]
    score_heads = [x[0] for x in sorted(zip(score_heads, head_distances), key=lambda x: x[1])]

    loc_heads = [x[0] for x in g.in_edges(locs_concat)]
    head_distances = [nx.algorithms.shortest_path_length(g, inp, x) for x in loc_heads]
    loc_heads = [x[0] for x in sorted(zip(loc_heads, head_distances), key=lambda x: x[1])]

    source_layers = []
    for score_head, loc_head in zip(score_heads, loc_heads):
        score_path = nx.algorithms.shortest_path(g, inp, score_head)
        loc_path = nx.algorithms.shortest_path(g, inp, loc_head)
        intersecion = [x for x, y in zip(score_path, loc_path) if x == y]
        source_layers.append(intersecion[-1])

    detector_cfg = config.model['detector']
    num_scales = len(source_layers)

    last_id = max(int(x['id']) for x in layers)

    if 'min_scale' in detector_cfg and 'max_scale' in detector_cfg:
        assert config.input_size[0] == config.input_size[1]
        scales = np.linspace(detector_cfg['min_scale'], detector_cfg['max_scale'], num_scales + 1)
        sizes = scales * config.input_size[0]
    elif 'sizes' in detector_cfg:
        assert len(detector_cfg['sizes']) == num_scales + 1
        sizes = detector_cfg['sizes']
    else:
        raise KeyError('Either size or scale should be specified')

    variance = ','.join(operator.add(*[[str(1 / float(config.box_coder[x]))] * 2 for x in ['xy_scale', 'wh_scale']]))
    num_branches = detector_cfg.get('num_branches', [1] * num_scales)

    prior_boxes = []
    prior_box_out_sizes = []
    for i, (source, ar, nb, loc_head) in enumerate(zip(source_layers, detector_cfg['aspect_ratios'], num_branches, loc_heads)):
        _ar = ','.join(str(x) for x in ar if x > 1.0)
        _sizes = np.linspace(sizes[i], sizes[i + 1], nb + 1)
        min_size = ','.join(str(x) for x in _sizes[:-1])
        max_size = ','.join(str(x) for x in _sizes[1:])

        port = layers_by_id[source].find('output').find_all('port')
        assert len(port) == 1
        port = port[0]
        dim = [x.text for x in port.find_all('dim')]
        assert len(dim) == 4

        loc_head_port = layers_by_id[loc_head].find('output').find_all('port')
        assert len(loc_head_port) == 1
        loc_head_dim = [x.text for x in loc_head_port[0].find_all('dim')]
        assert len(loc_head_dim) == 2
        prior_box_out_sizes.append(loc_head_dim[1])

        last_id += 1

        prior_box = PRIOR_BOX(
            id=last_id,
            i=i,
            aspect_ratio=_ar,
            min_size=min_size,
            max_size=max_size,
            variance=variance,
            num_channel=dim[1],
            filter_height=dim[2],
            filter_width=dim[3],
            input_width=config.input_size[0],
            input_height=config.input_size[1],
            size_boxes=loc_head_dim[1]
        )
        model.layers.insert(-1, prior_box)
        prior_boxes.append(last_id)

        model.edges.insert(-1, EDGE(inp, '0', str(last_id), '1'))
        model.edges.insert(-1, EDGE(source, port['id'], str(last_id), '0'))

    last_id += 1

    in_ports = []
    for i, (pb, s) in enumerate(zip(prior_boxes, prior_box_out_sizes)):
        in_ports.append(str(PORT(id=str(i), dims=DIMS('1', '2', s))))
        model.edges.insert(-1, EDGE(pb, '2', str(last_id), str(i)))

    pb_total_size = str(sum(int(x) for x in prior_box_out_sizes))
    out_port = PORT(id=len(prior_boxes), dims=DIMS('1', '2', pb_total_size))

    concat = CONCAT(id=str(last_id), input_ports=''.join(in_ports), output_ports=out_port)
    pb_output = concat.find('layer')['id']
    model.layers.insert(-1, concat)

    last_id += 1

    score_output_port = layers_by_id[score_output].select('output > port')
    assert len(score_output_port) == 1
    score_output_port = score_output_port[0]
    score_output_size = score_output_port.select('dim')[-1].text

    locs_output_port = layers_by_id[locs_output].select('output > port')
    assert len(locs_output_port) == 1
    locs_output_port = locs_output_port[0]
    locs_output_size = locs_output_port.select('dim')[-1].text

    assert pb_total_size == locs_output_size

    detection_output = DETECTION_OUTPUT(
        id=last_id,
        score_threshold=config.postprocess['score_threshold'],
        max_total=config.postprocess['max_total'],
        max_per_class=config.postprocess['nms']['max_per_class'],
        overlap_threshold=config.postprocess['nms']['overlap_threshold'],
        num_classes=detector_cfg['num_classes'],
        size_locs=locs_output_size,
        size_classes=score_output_size,
        size_boxes=pb_total_size
    )
    model.layers.insert(-1, detection_output)

    model.edges.insert(-1, EDGE(locs_output, locs_output_port['id'], str(last_id), '0'))
    model.edges.insert(-1, EDGE(score_output, score_output_port['id'], str(last_id), '1'))
    model.edges.insert(-1, EDGE(pb_output, str(len(prior_boxes)), str(last_id), '2'))

    parser = etree.XMLParser(remove_blank_text=True)
    with open(model_file, 'w+') as f:
        f.write(etree.tostring(etree.fromstring(model.decode_contents(), parser=parser), pretty_print=True).decode())

if __name__ == '__main__':
    model, config = sys.argv[1:3]
    config = helpers.load_config(config)
    add_output(model, config)
