"""
Source file with all the Gluon->PyTorch convertation functions.
"""

import torch
import numpy as np


def convert_conv(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    """
    Convert Conv2d layer.
    """

    assert(op['attrs']['layout'] == 'NCHW')

    init_tmp = ' ' * 8 +\
        'self.x{i} = nn.Conv2d(' +\
        '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={strides}, ' +\
        'bias={use_bias}, groups={num_group}, padding={padding})'
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


def convert_deconvolution(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    """
    Convert Conv2dTranspose layer.
    """

    assert(op['attrs']['layout'] == 'NCHW')

    init_tmp = ' ' * 8 +\
        'self.x{i} = nn.ConvTranspose2d(' +\
        '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={strides}, ' +\
        'bias={use_bias}, groups={num_group}, padding={padding})'
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
        'in_channels': weights.shape[0],
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


def convert_softmax(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = F.softmax(x{inp}, dim=len(x.size()) - 1)'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name
    })

    return '', call_str


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
            init_tmp = ' ' * 8 +\
                'self.x{i} = nn.MaxPool2d(' +\
                'kernel_size={kernel_size}, stride={stride}, padding={padding}, ' +\
                'ceil_mode={ceil_mode})'
            call_tmp = ' ' * 8 +\
                'x{i} = self.x{i}(x{inp})'
        elif op['attrs']['pool_type'] == 'avg':
            init_tmp = ' ' * 8 +\
                'self.x{i} = nn.AvgPool2d(' +\
                'kernel_size={kernel_size}, stride={stride}, padding={padding}, ' +\
                'count_include_pad={count_include_pad}, ceil_mode={ceil_mode})'
            call_tmp = ' ' * 8 +\
                'x{i} = self.x{i}(x{inp})'
        else:
            raise 'Unknown pooling'

        init_str = init_tmp.format(**{
            'i': i,
            'kernel_size': op['attrs']['kernel'],
            'stride': op['attrs']['stride'],
            'padding': op['attrs']['pad'],
            'ceil_mode': op['attrs']['pooling_convention'] == 'full',
            'count_include_pad': op['attrs']['count_include_pad'] if 'count_include_pad' in op['attrs'] else 'True',
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


def convert_elemwise_sub(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = x{l} - x{r}'

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


def convert_elemwise_mul(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = x{l} * x{r}'

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
    call_tmp = ' ' * 8 + 'x{i} = torch.cat([{inputs}], dim={dim})'

    call_str = call_tmp.format(**{
        'i': i,
        'inputs': ','.join(['x{0}'.format(i) for i in op['inputs']]),
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


def convert_slice(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = x{l}[{slices}]'

    op['attrs']['begin'] = eval(op['attrs']['begin'])
    op['attrs']['end'] = eval(op['attrs']['end'])

    begins = [str(i) if i != None else '' for i in list(op['attrs']['begin'])]
    ends = [str(i) if i != None else '' for i in list(op['attrs']['end'])]

    slices = ','.join([':'.join(i) for i in zip(begins, ends)])

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'slices': slices,
    })

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

    return '', call_str


def convert_adaptive_avg_pool(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = F.adaptive_avg_pool2d(x{l}, {size})'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i,
        'l': input_names[0],
        'size': op['attrs']['output_size'],
    })

    return '', call_str


def convert_dropout(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
    init_tmp = ' ' * 8 + 'self.x{i} = nn.Dropout(p={p})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i,
        'p': op['attrs']['p'],
    })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name,
    })

    print(init_str, call_str)
    return init_str, call_str


def convert_leaky_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'

    if op['attrs']['act_type'] == 'selu':
        init_tmp = ' ' * 8 + 'self.x{i} = nn.SELU()'
    elif op['attrs']['act_type'] == 'leaky':
        init_tmp = ' ' * 8 + 'self.x{i} = nn.LeakyReLU(negative_slope={slope})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    if op['attrs']['act_type'] == 'selu':
        init_str = init_tmp.format(**{
            'i': i,
        })
    else:
        init_str = init_tmp.format(**{
            'i': i,
            'slope': op['attrs']['slope'],
        })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name,
    })

    print(init_str, call_str)
    return init_str, call_str


def convert_pad(i, op, gluon_nodes, gluon_dict, pytorch_dict):
    call_tmp = ' ' * 8 + 'x{i} = self.x{i}(x{inp})'
    if op['attrs']['mode'] == 'reflect':
        init_tmp = ' ' * 8 + 'self.x{i} = nn.ReflectionPad2d(padding={padding})'
    elif op['attrs']['mode'] == 'edge': 
        init_tmp = ' ' * 8 + 'self.x{i} = nn.ReplicationPad2d(padding={padding})'
    elif op['attrs']['mode'] == 'constant':
        init_tmp = ' ' * 8 + 'self.x{i} = nn.ConstantPad2d(padding={padding}, value={constant_value})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    op['attrs']['pad_width'] = eval(op['attrs']['pad_width'])

    if np.sum(list(op['attrs']['pad_width'])[:4]) > 0:
        raise 'Not implemented batch/channel axis padding'

    if op['attrs']['mode'] == 'constant':
        init_str = init_tmp.format(**{
            'i': i,
            'constant_value': op['attrs']['constant_value'] if 'constant_value' in op['attrs'] else 0,
            'padding': op['attrs']['pad_width'][4:],
        })
    else:
        init_str = init_tmp.format(**{
            'i': i,
            'padding': op['attrs']['pad_width'][4:],
        })

    call_str = call_tmp.format(**{
        'i': i,
        'inp': input_name,
    })

    print(init_str, call_str)
    return init_str, call_str


# Here will be converters.
CONVERTERS = {
    'Activation': convert_activation,
    'BatchNorm': convert_batchnorm,
    'Concat': convert_concat,
    'Convolution': convert_conv,
    'Flatten': convert_flatten,
    'FullyConnected': convert_linear,
    'Pooling': convert_pooling,
    'Dropout': convert_dropout,
    'relu': convert_relu,
    'sigmoid': convert_sigmoid,
    'softmax': convert_softmax,
    'elemwise_add': convert_elemwise_add,
    'elemwise_sub': convert_elemwise_sub,
    'elemwise_mul': convert_elemwise_mul,
    'slice_axis': convert_slice_axis,
    'slice': convert_slice,
    '_mul_scalar': convert_mul_scalar,
    '_contrib_AdaptiveAvgPooling2D': convert_adaptive_avg_pool,
    'broadcast_mul': convert_elemwise_mul,
    'LeakyReLU': convert_leaky_relu,
    'Pad': convert_pad,
    'Deconvolution': convert_deconvolution,
}
