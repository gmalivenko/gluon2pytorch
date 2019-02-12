"""
Source file with all the Gluon->PyTorch convertation functions.
"""

import torch
import numpy as np


def transform_names(i, op, names_dict):
    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = ['x' + str(i) if names_dict is None else names_dict[i] for i in op['inputs']]

    output_name = 'x' + str(i) if names_dict is None else names_dict[i]

    return input_names, output_name


def convert_conv(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    """
    Convert Conv2d layer.
    """

    assert(op['attrs']['layout'] == 'NCHW')

    input_names, output_name = transform_names(i, op, names_dict)

    init_tmp = ' ' * 8 +\
        'self.{i} = nn.Conv2d(' +\
        '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={strides}, dilation={dilation}, ' +\
        'bias={use_bias}, groups={num_group}, padding={padding})'
    call_tmp = ' ' * 8 +\
        '{i} = self.{i}({inp})'

    weights = gluon_dict[op['name'] + '_weight'].data().asnumpy()

    pytorch_dict['{i}.weight'.format(i=output_name)] = torch.FloatTensor(weights)

    bias = None
    use_bias = not eval(op['attrs']['no_bias'])

    if use_bias:
        bias = gluon_dict[op['name'] + '_bias'].data().asnumpy()
        pytorch_dict['{i}.bias'.format(i=output_name)] = torch.FloatTensor(bias)

    init_str = init_tmp.format(**{
        'i': output_name,
        'in_channels': weights.shape[1] * int(op['attrs']['num_group']),
        'out_channels': int(op['attrs']['num_filter']),
        'use_bias': use_bias,
        'kernel_size': op['attrs']['kernel'],
        'strides': op['attrs']['stride'],
        'padding': op['attrs']['pad'],
        'dilation': op['attrs']['dilate'],
        'num_group': op['attrs']['num_group']
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_deconvolution(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    """
    Convert Conv2dTranspose layer.
    """

    assert(op['attrs']['layout'] == 'NCHW')

    input_names, output_name = transform_names(i, op, names_dict)

    init_tmp = ' ' * 8 +\
        'self.{i} = nn.ConvTranspose2d(' +\
        '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={strides}, ' +\
        'bias={use_bias}, groups={num_group}, padding={padding})'
    call_tmp = ' ' * 8 +\
        '{i} = self.{i}({inp})'

    weights = gluon_dict[op['name'] + '_weight'].data().asnumpy()

    pytorch_dict['{i}.weight'.format(i=output_name)] = torch.FloatTensor(weights)

    bias = None
    use_bias = not eval(op['attrs']['no_bias'])

    if use_bias:
        bias = gluon_dict[op['name'] + '_bias'].data().asnumpy()
        pytorch_dict['{i}.bias'.format(i=output_name)] = torch.FloatTensor(bias)

    init_str = init_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'in_channels': weights.shape[0],
        'out_channels': int(op['attrs']['num_filter']),
        'use_bias': use_bias,
        'kernel_size': op['attrs']['kernel'],
        'strides': op['attrs']['stride'],
        'padding': op['attrs']['pad'],
        'num_group': op['attrs']['num_group']
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    init_tmp = ' ' * 8 + 'self.{i} = nn.ReLU()'

    input_names, output_name = transform_names(i, op, names_dict)

    init_str = init_tmp.format(**{
        'i': output_name,
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_sigmoid(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    init_tmp = ' ' * 8 + 'self.{i} = nn.Sigmoid()'

    input_names, output_name = transform_names(i, op, names_dict)

    init_str = init_tmp.format(**{
        'i': output_name,
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_softmax(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = F.softmax({inp}, dim=len(x.size()) - 1)'

    input_names, output_name = transform_names(i, op, names_dict)

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return '', call_str


def convert_batchnorm(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    init_tmp = ' ' * 8 + 'self.{i} = nn.BatchNorm2d({in_channels}, momentum={momentum}, eps={eps})'
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'

    input_names, output_name = transform_names(i, op, names_dict)

    gamma = gluon_dict[op['name'] + '_gamma'].data().asnumpy()
    beta = gluon_dict[op['name'] + '_beta'].data().asnumpy()
    running_mean = gluon_dict[op['name'] + '_running_mean'].data().asnumpy()
    running_var = gluon_dict[op['name'] + '_running_var'].data().asnumpy()

    pytorch_dict['{i}.weight'.format(i=output_name)] = torch.FloatTensor(gamma)
    pytorch_dict['{i}.bias'.format(i=output_name)] = torch.FloatTensor(beta)

    pytorch_dict['{i}.running_mean'.format(i=output_name)] = torch.FloatTensor(running_mean)
    pytorch_dict['{i}.running_var'.format(i=output_name)] = torch.FloatTensor(running_var)

    init_str = init_tmp.format(**{
        'i': output_name,
        'in_channels': gamma.shape[0],
        'momentum': op['attrs']['momentum'],
        'eps': op['attrs']['eps'],
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_activation(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    if op['attrs']['act_type'] == 'relu':
        return convert_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug)
    elif op['attrs']['act_type'] == 'sigmoid':
        return convert_sigmoid(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug)


def convert_pooling(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    input_names, output_name = transform_names(i, op, names_dict)

    if 'global_pool' in op['attrs'] and op['attrs']['global_pool'] == 'True':
        if op['attrs']['pool_type'] == 'max':
            init_tmp = ''
            call_tmp = ' ' * 8 + '{i} = F.adaptive_max_pool2d({inp}, (1, 1))'
        elif op['attrs']['pool_type'] == 'avg':
            init_tmp = ''
            call_tmp = ' ' * 8 + '{i} = F.adaptive_avg_pool2d({inp}, (1, 1))'
        else:
            raise 'Unknown pooling'

        init_str = ''

        call_str = call_tmp.format(**{
            'i': output_name,
            'inp': input_names[0]
        })

    else:
        if op['attrs']['pool_type'] == 'max':
            init_tmp = ' ' * 8 +\
                'self.{i} = nn.MaxPool2d(' +\
                'kernel_size={kernel_size}, stride={stride}, padding={padding}, ' +\
                'ceil_mode={ceil_mode})'
            call_tmp = ' ' * 8 +\
                '{i} = self.{i}({inp})'
        elif op['attrs']['pool_type'] == 'avg':
            init_tmp = ' ' * 8 +\
                'self.{i} = nn.AvgPool2d(' +\
                'kernel_size={kernel_size}, stride={stride}, padding={padding}, ' +\
                'count_include_pad={count_include_pad}, ceil_mode={ceil_mode})'
            call_tmp = ' ' * 8 +\
                '{i} = self.{i}({inp})'
        else:
            raise 'Unknown pooling'

        init_str = init_tmp.format(**{
            'i': output_name,
            'kernel_size': op['attrs']['kernel'],
            'stride': op['attrs']['stride'],
            'padding': op['attrs']['pad'] if 'pad' in op['attrs'] else [0, 0],
            'ceil_mode': op['attrs']['pooling_convention'] == 'full',
            'count_include_pad': op['attrs']['count_include_pad'] if 'count_include_pad' in op['attrs'] else 'True',
        })

        call_str = call_tmp.format(**{
            'i': output_name,
            'inp': input_names[0]
        })

    return init_str, call_str


def convert_elemwise_add(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l} + {r}'

    input_names, output_name = transform_names(i, op, names_dict)

    call_str = call_tmp.format(**{
        'i': output_name,
        'l': input_names[0],
        'r': input_names[1],
    })

    return '', call_str


def convert_elemwise_sub(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l} - {r}'

    input_names, output_name = transform_names(i, op, names_dict)

    call_str = call_tmp.format(**{
        'i': output_name,
        'l': input_names[0],
        'r': input_names[1],
    })

    return '', call_str


def convert_elemwise_mul(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l} * {r}'

    input_names, output_name = transform_names(i, op, names_dict)

    call_str = call_tmp.format(**{
        'i': output_name,
        'l': input_names[0],
        'r': input_names[1],
    })

    return '', call_str


def convert_flatten(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.view([int({l}.size(0)), -1])'

    input_names, output_name = transform_names(i, op, names_dict)

    call_str = call_tmp.format(**{
        'i': output_name,
        'l': input_names[0],
    })

    return '', call_str


def convert_linear(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    init_tmp = ' ' * 8 + 'self.{i} = nn.Linear({in_channels}, {out_channels}, bias={use_bias})'

    input_names, output_name = transform_names(i, op, names_dict)

    if op['attrs']['flatten'] == 'True':
        call_tmp = ' ' * 8 + '{i} = self.{i}({inp}.view([int({inp}.size(0)), -1]))'
    else:
        call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'

    use_bias = not eval(op['attrs']['no_bias'])
    weights = gluon_dict[op['name'] + '_weight'].data().asnumpy()
    bias = None

    pytorch_dict['{i}.weight'.format(i=output_name)] = torch.FloatTensor(weights)
    if use_bias:
        bias = gluon_dict[op['name'] + '_bias'].data().asnumpy()
        pytorch_dict['{i}.bias'.format(i=output_name)] = torch.FloatTensor(bias)


    init_str = init_tmp.format(**{
        'i': output_name,
        'in_channels': weights.shape[1],
        'out_channels': weights.shape[0],
        'use_bias': use_bias,
    })

    call_str = call_tmp.format(**{
        'i': output_name,
        'inp': input_names[0]
    })

    return init_str, call_str


def convert_concat(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = torch.cat([{inputs}], dim={dim})'

    call_str = call_tmp.format(**{
        'i': i,
        'inputs': ','.join(['x{0}'.format(i) for i in op['inputs']]),
        'dim': op['attrs']['dim']
    })

    return '', call_str


def convert_slice_axis(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    if op['attrs']['axis'] == '0':
        call_tmp = ' ' * 8 + '{i} = {l}[{start}:{end}]'
    elif op['attrs']['axis'] == '1':
        call_tmp = ' ' * 8 + '{i} = {l}[:, {start}:{end}]'
    elif op['attrs']['axis'] == '2':
        call_tmp = ' ' * 8 + '{i} = {l}[:, :, {start}:{end}]'
    elif op['attrs']['axis'] == '3':
        call_tmp = ' ' * 8 + '{i} = {l}[:, :, :, {start}:{end}]'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'start': op['attrs']['begin'],
        'end': op['attrs']['end'] if op['attrs']['end'] != 'None' else '',
    })

    return '', call_str


def convert_slice(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}[{slices}]'

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
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'slices': slices,
    })

    return '', call_str


def convert_plus_scalar(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {scalar} + {l}'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'scalar': op['attrs']['scalar'],
    })

    return '', call_str


def convert_mul_scalar(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {scalar} * {l}'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'scalar': op['attrs']['scalar'],
    })

    return '', call_str


def convert_adaptive_avg_pool(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = F.adaptive_avg_pool2d({l}, {size})'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'size': op['attrs']['output_size'],
    })

    return '', call_str


def convert_dropout(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    init_tmp = ' ' * 8 + 'self.{i} = nn.Dropout(p={p})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'p': op['attrs']['p'],
    })

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'inp': input_name,
    })

    return init_str, call_str


def convert_leaky_relu(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'

    if op['attrs']['act_type'] == 'selu':
        init_tmp = ' ' * 8 + 'self.{i} = nn.SELU()'
    elif op['attrs']['act_type'] == 'leaky':
        init_tmp = ' ' * 8 + 'self.{i} = nn.LeakyReLU(negative_slope={slope})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    if op['attrs']['act_type'] == 'selu':
        init_str = init_tmp.format(**{
            'i': i if names_dict is None else names_dict[i],
        })
    else:
        init_str = init_tmp.format(**{
            'i': i if names_dict is None else names_dict[i],
            'slope': op['attrs']['slope'],
        })

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'inp': input_name,
    })

    return init_str, call_str


def convert_pad(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    if op['attrs']['mode'] == 'reflect':
        init_tmp = ' ' * 8 + 'self.{i} = nn.ReflectionPad2d(padding={padding})'
    elif op['attrs']['mode'] == 'edge': 
        init_tmp = ' ' * 8 + 'self.{i} = nn.ReplicationPad2d(padding={padding})'
    elif op['attrs']['mode'] == 'constant':
        init_tmp = ' ' * 8 + 'self.{i} = nn.ConstantPad2d(padding={padding}, value={constant_value})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    op['attrs']['pad_width'] = eval(op['attrs']['pad_width'])

    if np.sum(list(op['attrs']['pad_width'])[:4]) > 0:
        raise 'Not implemented batch/channel axis padding'

    if op['attrs']['mode'] == 'constant':
        init_str = init_tmp.format(**{
            'i': i if names_dict is None else names_dict[i],
            'constant_value': op['attrs']['constant_value'] if 'constant_value' in op['attrs'] else 0,
            'padding': op['attrs']['pad_width'][4:],
        })
    else:
        init_str = init_tmp.format(**{
            'i': i if names_dict is None else names_dict[i],
            'padding': op['attrs']['pad_width'][4:],
        })

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'inp': input_name,
    })

    return init_str, call_str


def convert_clip(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.clamp({min}, {max})'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'min': op['attrs']['a_min'],
        'max': op['attrs']['a_max'],
    })

    return '', call_str



def convert_reshape(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.contiguous().view({shape})'

    op['attrs']['shape'] = eval(op['attrs']['shape'])

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    pytorch_shape = []
    r = 0
    rr = 0
    while r < len(op['attrs']['shape']):
        k = op['attrs']['shape'][r]

        if k == 0:
            pytorch_shape.append('x{1}.size({0})'.format(rr, str(input_names[0])))
            rr += 1
        elif k >= 1 or k == -1:
            pytorch_shape.append(str(k))
            rr += 1
        elif k == -2:
            pytorch_shape.append('*(x{1}.size()[{0}:])'.format(rr, str(input_names[0])))
        elif k == -3:
            pytorch_shape.append('x{0}.size({1}) * x{0}.size({2})'.format(str(input_names[0]), rr, rr + 1))
            rr += 2
            # r += 1
        elif k == -4:
            pytorch_shape.append(str(op['attrs']['shape'][r + 1]))
            pytorch_shape.append(str('x{0}.size({1}) // {2}'.format(str(input_names[0]), rr, op['attrs']['shape'][r + 1])))
            rr += 1
            r += 1
        r += 1
        # rr += 1
    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'shape': ','.join([str(i) for i in pytorch_shape]),
    })

    return '', call_str


def convert_swap_axis(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.transpose({axis_a}, {axis_b})'


    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'axis_a': op['attrs']['dim1'],
        'axis_b': op['attrs']['dim2'],
    })

    return '', call_str


def convert_bilinear_resize2d(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    init_tmp = ' ' * 8 + 'self.{i} = nn.Upsample(size={size}, mode=\'bilinear\', align_corners=True)'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'size': (int(op['attrs']['height']), int(op['attrs']['width'])),
    })

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'inp': input_name,
    })

    return init_str, call_str


def convert_copy(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.clone()'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0]
    })

    return '', call_str


def convert_expand_dims(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.unsqueeze({dim})'

    if len(op['inputs']) == 0:
        input_names = ['']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'dim': eval(op['attrs']['axis'])
    })

    return '', call_str


def convert_broadcast_like(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.view(x{s}.size())'

    if len(op['inputs']) == 0:
        input_names = ['', '']
    else:
        input_names = [str(op['inputs'][0]), str(op['inputs'][1])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        's': input_names[1],
    })

    return '', call_str


def convert_sum(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.sum({dim})'

    if len(op['inputs']) == 0:
        input_names = ['', '']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'dim': eval(op['attrs']['axis']),
    })

    return '', call_str


def convert_max(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.max({dim})'

    if len(op['inputs']) == 0:
        input_names = ['', '']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'dim': eval(op['attrs']['axis']),
    })

    return '', call_str


def convert_mean(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = {l}.mean({dim})'

    if len(op['inputs']) == 0:
        input_names = ['', '']
    else:
        input_names = [str(op['inputs'][0])]

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'l': input_names[0],
        'dim': eval(op['attrs']['axis']),
    })

    return '', call_str


def convert_lrn(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    call_tmp = ' ' * 8 + '{i} = self.{i}({inp})'
    init_tmp = ' ' * 8 + 'self.{i} = nn.LocalResponseNorm(size={size}, k={knorm}, alpha={alpha}, beta={beta})'

    if len(op['inputs']) == 0:
        input_name = ''
    else:
        input_name = op['inputs'][0]

    init_str = init_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'size': eval(op['attrs']['nsize']),
        'knorm': eval(op['attrs']['knorm']) if 'knorm' in op['attrs'] else 2.0,
        'alpha': eval(op['attrs']['alpha']) if 'alpha' in op['attrs'] else 0.0001,
        'beta': eval(op['attrs']['beta']) if 'beta' in op['attrs'] else 0.75
    })

    call_str = call_tmp.format(**{
        'i': i if names_dict is None else names_dict[i],
        'inp': input_name
    })

    return init_str, call_str


def convert_upsampling(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug):
    if op['attrs']['sample_type'] == 'nearest':
        call_tmp = ' ' * 8 + '{i} = F.upsample_nearest({l}, scale_factor={scale})'

        if len(op['inputs']) == 0:
            input_names = ['', '']
        else:
            input_names = [str(op['inputs'][0])]

        call_str = call_tmp.format(**{
            'i': i if names_dict is None else names_dict[i],
            'l': input_names[0],
            'scale': eval(op['attrs']['scale']),
        })

        return '', call_str
    else:
        return convert_bilinear_resize2d(i, op, gluon_nodes, gluon_dict, pytorch_dict, names_dict, debug)


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
    '_plus_scalar': convert_plus_scalar,
    '_mul_scalar': convert_mul_scalar,
    '_contrib_AdaptiveAvgPooling2D': convert_adaptive_avg_pool,
    'broadcast_mul': convert_elemwise_mul,
    'LeakyReLU': convert_leaky_relu,
    'Pad': convert_pad,
    'Deconvolution': convert_deconvolution,
    'clip': convert_clip,
    'Reshape': convert_reshape,
    'SwapAxis': convert_swap_axis,
    '_contrib_BilinearResize2D': convert_bilinear_resize2d,
    '_copy': convert_copy,
    'expand_dims': convert_expand_dims,
    'broadcast_like': convert_broadcast_like,
    'sum': convert_sum,
    'max': convert_max,
    'mean': convert_mean,
    'LRN': convert_lrn,
    'UpSampling': convert_upsampling,
}
