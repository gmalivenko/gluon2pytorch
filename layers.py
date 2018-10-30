"""
Source file with all the Gluon->PyTorch convertation functions.
"""

import numpy as np
import torch



def convert_conv(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    """
    Convert Conv2d layer.
    """

    assert(op['attrs']['layout'] == 'NCHW')

    init_tmp = ' ' * 8 +\
        'self.x{i} = nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={strides}, bias={use_bias}, groups={num_group}, padding={padding})'
    call_tmp = ' ' * 8 +\
        'x{i} = self.x{i}(x{inp})'

    weights = gluon_dict[op['name'] + '_weight'].data().asnumpy()
   
    pytorch_dict['x{i}.weight'.format(i=i)] = torch.FloatTensor(weights)

    bias = None
    use_bias = not bool(op['attrs']['no_bias'])
    if use_bias:
        bias = gluon_dict[op['name'] + '_bias'].data.asnumpy()
        pytorch_dict['x{i}.bias'.format(i=i)] = torch.FloatTensor(bias)

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
        'in_channels': weights.shape[1] * int(op['attrs']['num_group']),
        'out_channels': int(op['attrs']['num_filter']),
        'use_bias': use_bias,
        'kernel_size': op['attrs']['kernel'],
        'strides': op['attrs']['stride'],
        'padding': op['attrs']['pad'],
        'num_group': op['attrs']['num_group']
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })
    
    return init_str, call_str
    

def convert_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
    init_tmp = ' ' * 8 + 'self.x{i} = nn.ReLU()'
    
    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })

    print(init_str, call_str)
    return init_str, call_str


def convert_sigmoid(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
    init_tmp = ' ' * 8 + 'self.x{i} = nn.Sigmoid()'
    
    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })

    print(init_str, call_str)
    return init_str, call_str


def convert_batchnorm(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    init_tmp = ' ' * 8 + 'self.x{i} = nn.BatchNorm2d({in_channels}, momentum={momentum}, eps={eps})'
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'

    gamma = gluon_dict[op['name'] + '_gamma'].data().asnumpy()
    beta = gluon_dict[op['name'] + '_beta'].data().asnumpy()
    running_mean = gluon_dict[op['name'] + '_running_mean'].data().asnumpy()
    running_var = gluon_dict[op['name'] + '_running_var'].data().asnumpy()

    pytorch_dict['x{i}.weight'.format(i=i)] = torch.FloatTensor(gamma) 
    pytorch_dict['x{i}.bias'.format(i=i)] = torch.FloatTensor(beta)

    pytorch_dict['x{i}.running_mean'.format(i=i)] = torch.FloatTensor(running_mean)  
    pytorch_dict['x{i}.running_var'.format(i=i)] = torch.FloatTensor(running_var) 

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
        'in_channels': gamma.shape[0],
        'momentum': op['attrs']['momentum'],
        'eps': op['attrs']['eps'],
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })

    print(init_str, call_str)
    return init_str, call_str


def convert_activation(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    if op['attrs']['act_type'] == 'relu':
        return convert_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict)
    elif op['attrs']['act_type'] == 'sigmoid':
        return convert_sigmoid(i, op, gluon_nodes, gluon_dict, pytorch_dict)

    
# Here will be converters.
CONVERTERS = {
    'Activation': convert_activation,
    'BatchNorm': convert_batchnorm,
    'Convolution': convert_conv,
    'relu': convert_relu,
    'sigmoid': convert_sigmoid,
}