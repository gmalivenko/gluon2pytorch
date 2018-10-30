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
    

# Here will be converters.
CONVERTERS = {
	'Convolution': convert_conv,
}