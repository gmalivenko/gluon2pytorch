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


def convert_pooling(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    # {'name': 'resnet18_v1_pool0', 'type': 'Pooling', 'inputs': array([2]),
    #'attrs': {'global_pool': 'False', 'kernel': '(3, 3)', 'pad': '(1, 1)', 'pool_type': 'max', 'pooling_convention': 'valid', 'stride': '(2, 2)'}}

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    if op['attrs']['global_pool'] == 'True':
        if op['attrs']['pool_type'] == 'max':
            init_tmp = ''
            call_tmp = ' ' * 8 + 'x{i} = F.adaptive_max_pool2d(x{inp}, (1, 1))'
        elif op['attrs']['pool_type'] == 'avg':
            init_tmp = ''
            call_tmp = ' ' * 8 + 'x{i} = F.adaptive_avg_pool2d(x{inp}, (1, 1))'
        else:
            raise 'Unknown pooling'

        init_str = ''

        call_str = call_tmp.format(**{
            'i': i,
            'inp': input_name
        })

    else:
        if op['attrs']['pool_type'] == 'max':
            init_tmp = ' ' * 8 + 'self.x{i} = nn.MaxPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})'
            call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
        elif op['attrs']['pool_type'] == 'avg':
            init_tmp = ' ' * 8 + 'self.x{i} = nn.AvgPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})'
            call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
        else:
            raise 'Unknown pooling'

    

        init_str = init_tmp.format(**{
            'i': i,
            'kernel_size': op['attrs']['kernel'],
            'stride': op['attrs']['stride'],
            'padding': op['attrs']['pad'],
        })

        call_str = call_tmp.format(**{
            'i': i,
            'inp': input_name
        })

    print(init_str, call_str)
    return init_str, call_str


def convert_elemwise_add(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = x{l} + x{r}'

    if len(op['inputs']) == 1:
        input_names = ['', str(op['inputs'][0])]
    else:
        input_names = [str(op['inputs'][0]), str(op['inputs'][1])]

    
    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'r': input_names[1],
    })

    print(call_str)
    return '', call_str


def convert_flatten(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = x{l}.view(x{l}.size(0), -1)'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
    })

    print(call_str)
    return '', call_str


def convert_linear(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    init_tmp = ' ' * 8 + 'self.x{i} = nn.Linear({in_channels}, {out_channels}, bias={use_bias})'

    if op['attrs']['flatten'] == 'True':
        call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp}.view(x{inp}.size(0), -1))'
    else:
        call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'

    
    use_bias = not bool(op['attrs']['no_bias'])
    weights = gluon_dict[op['name'] + '_weight'].data().asnumpy()
    bias = None

    pytorch_dict['x{i}.weight'.format(i=i)] = torch.FloatTensor(weights)
    if use_bias:
        bias = gluon_dict[op['name'] + '_bias'].data.asnumpy()
        pytorch_dict['x{i}.bias'.format(i=i)] = torch.FloatTensor(bias)

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
        'in_channels': weights.shape[1],
        'out_channels': weights.shape[0],
        'use_bias': use_bias,
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })
    
    print(init_str, call_str)
    return init_str, call_str



def convert_concat(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = torch.cat([x{l}, x{r}], dim={dim})'

    if len(op['inputs']) == 1:
        input_names = ['', str(op['inputs'][0])]
    else:
        input_names = [str(op['inputs'][0]), str(op['inputs'][1])]
    
    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'r': input_names[1],
        'dim': op['attrs']['dim']
    })

    print(call_str)
    return '', call_str


def convert_slice_axis(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    if op['attrs']['axis'] == '0':
        call_tmp = ' ' * 8 + 'x{i} = x{l}[{start}:{end}]'
    elif op['attrs']['axis'] == '1':
        call_tmp = ' ' * 8 + 'x{i} = x{l}[:, {start}:{end}]'
    elif op['attrs']['axis'] == '2':
        call_tmp = ' ' * 8 + 'x{i} = x{l}[:, :, {start}:{end}]'
    elif op['attrs']['axis'] == '3':
        call_tmp = ' ' * 8 + 'x{i} = x{l}[:, :, :, {start}:{end}]'
     
    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]
    
    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'start': op['attrs']['begin'],
        'end': op['attrs']['end'] if op['attrs']['end'] != 'None' else '',
    })

    print(call_str)
    return '', call_str


def convert_mul_scalar(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = {scalar} * x{l}'
     
    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]
    
    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'scalar': op['attrs']['scalar'],
    })

    print(call_str)
    return '', call_str


# Here will be converters.
CONVERTERS = {
    'Activation': convert_activation,
    'BatchNorm': convert_batchnorm,
    'Concat': convert_concat,
    'Convolution': convert_conv,
    'Flatten': convert_flatten,
    'FullyConnected': convert_linear,
    'Pooling': convert_pooling,
    'relu': convert_relu,
    'sigmoid': convert_sigmoid,
    'elemwise_add': convert_elemwise_add,
    'slice_axis': convert_slice_axis,
    '_mul_scalar': convert_mul_scalar,
}