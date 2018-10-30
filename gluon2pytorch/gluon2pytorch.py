import json
import os

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
    exec(pytorch_source)
    globals()[module_name] = locals()[module_name]
    pytorch_model = locals()[module_name]()
    pytorch_model.load_state_dict(pytorch_dict)
    return pytorch_model


def render_module(inits, calls, dst_dir, pytorch_dict, pytorch_module_name):
    """
    Render model.
    """

    output = pytorch_model_template.format(**{
        'module_name': pytorch_module_name,
        'module_name_lower': pytorch_module_name.lower(),
        'inits': '\n'.join(inits),
        'calls': '\n'.join(calls),
        'last': len(calls) - 1
    })

    if dst_dir is not None:
        import os, errno

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


def gluon2pytorch(net, dst_dir, pytorch_module_name, debug=True):
    """
    Function to convert a model.
    """

    x = net(mx.nd.array(np.ones((1, 3, 224, 224))))
    # print(x)

    # Get network params
    params = net.collect_params()

    # Create a symbol to trace net
    x = mx.sym.var('data')
    sym = net(x)
    
    # Get JSON-definition of the model    
    json_model = json.loads(sym.tojson())['nodes']

    # Create empty accumulators
    nodes = []
    is_skipped = []
    pytorch_dict = {}
    inits = []
    calls = []

    # Trace model
    for i, node in enumerate(json_model):
        # If the node has 'null' op, it means, that it's not a real op, but only parameter
        # TODO: convert constants
        if node['op'] == 'null':
            is_skipped.append(1)
            continue

        # It's not 'null'
        is_skipped.append(0)
        
        # Create dict with necessary node parameters        
        op = {
            'name': node['name'][:-4],
            'type': node['op'],
        }

        try:
            # Not all nodes have 'inputs'
            op['inputs'] = [i - np.sum(is_skipped[:i]) for i in np.array(node['inputs'])[:, 0] if is_skipped[i] != 1]
            op['inputs'] = np.array(op['inputs'])
        except:
            op['inputs'] = []

        try:
            # Not all nodes have 'attrs'
            op['attrs'] = node['attrs']
        except:
            op['attrs'] = {}

        # Debug output
        if debug:
            print(op)
            print('__')

        # Append new node to list
        nodes.append(op)

        # If operation is in available convertors, convert it
        if op['type'] in CONVERTERS:
            init_str, call_str = CONVERTERS[op['type']](i - np.sum(is_skipped[:i]), op, nodes, params, pytorch_dict)
            inits.append(init_str)
            calls.append(call_str)
        else:
            raise AttributeError('Layer isnt supported')

    pytorch_source = render_module(inits, calls, dst_dir, pytorch_dict, pytorch_module_name)

    return eval_model(pytorch_source, pytorch_dict, pytorch_module_name)