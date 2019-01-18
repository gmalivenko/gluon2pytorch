import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import mxnet as mx
import numpy as np

# Import comverters
from .layers import CONVERTERS

# Import PyTorch model template
from .pytorch_model_template import pytorch_model_template


def eval_model(pytorch_source, pytorch_dict, module_name):
    # Tricky code
    torch
    nn
    F
    exec(pytorch_source)
    globals()[module_name] = locals()[module_name]
    pytorch_model = locals()[module_name]()
    pytorch_model.load_state_dict(pytorch_dict)
    return pytorch_model


def render_module(inits, calls, inputs, outputs, dst_dir, pytorch_dict, pytorch_module_name):
    """
    Render model.
    """

    inits = [i for i in inits if len(i) > 0]

    output = pytorch_model_template.format(**{
        'module_name': pytorch_module_name,
        'module_name_lower': pytorch_module_name.lower(),
        'inits': '\n'.join(inits),
        'inputs': ', '.join(['x' + str(i) for i in inputs]),
        'calls': '\n'.join(calls),
        'outputs': ', '.join(['x' + str(i) for i in outputs]),
    })

    if dst_dir is not None:
        import os
        import errno

        try:
            os.makedirs(dst_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(os.path.join(dst_dir, pytorch_module_name.lower() + '.py'), 'w+') as f:
            f.write(output)
            f.close()

        torch.save(pytorch_dict, os.path.join(dst_dir, pytorch_module_name.lower() + '.pt'))

    return output


def gluon2pytorch(net, args, dst_dir, pytorch_module_name, debug=True):
    """
    Function to convert a model.
    """

    x = [mx.nd.array(np.ones(i)) for i in args]
    x = net(*x)

    # Get network params
    params = net.collect_params()

    # Create a symbol to trace net
    # x = mx.sym.var('data')
    x = [mx.sym.var('__input__' + str(i)) for i in range(len(args))]
    sym = net(*x)

    if len(sym) > 1:
        group = mx.sym.Group(sym)
    else:
        group = sym

    # Get JSON-definition of the model
    json_model = json.loads(group.tojson())['nodes']

    # Create empty accumulators
    nodes = []
    is_skipped = []
    pytorch_dict = {}
    inits = []
    calls = []
    inputs = []
    outputs = [i[0] for i in json.loads(group.tojson())['heads']] 
    last = 0

    # Trace model
    for i, node in enumerate(json_model):
        # If the node has 'null' op, it means, that it's not a real op, but only parameter
        # TODO: convert constants
        if node['op'] == 'null':
            if node['name'].find('__input__') == 0:
                inputs.append(int(node['name'][9:]))
            is_skipped.append(1)
            continue

        # It's not 'null'
        is_skipped.append(0)

        # Create dict with necessary node parameters
        op = {
            'name': node['name'][:-4],
            'type': node['op'],
        }
        print(op, node)
        if len(node['inputs']) > 0:
            orginal_inputs = [i for i in np.array(node['inputs'])[:, 0] if i in inputs]
            op['inputs'] = [i for i in np.array(node['inputs'])[:, 0] if is_skipped[i] != 1 or i in orginal_inputs]
        else:
            print(json_model)
            op['inputs'] = []
        try:
            # Not all nodes have 'attrs'
            op['attrs'] = node['attrs']
        except KeyError:
            op['attrs'] = {}

        # Debug output
        if debug:
            print(op)
            print('__')

        # Append new node to list
        nodes.append(op)

        # If operation is in available convertors, convert it
        if op['type'] in CONVERTERS:
            init_str, call_str = CONVERTERS[op['type']](i, op, nodes, params, pytorch_dict)
            inits.append(init_str)
            calls.append(call_str)
        else:
            raise AttributeError('Layer isn\'t supported')

    pytorch_source = render_module(inits, calls, inputs, outputs, dst_dir, pytorch_dict, pytorch_module_name)

    return eval_model(pytorch_source, pytorch_dict, pytorch_module_name)
