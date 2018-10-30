import argparse
import torch
import mxnet as mx
import numpy as np
import sys

sys.path.append('../')
from gluon2pytorch import gluon2pytorch


class Conv2dTest(mx.gluon.nn.HybridSequential):
    def __init__(self, filters=64, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, padding_h=1, padding_w=1, use_bias=False):
        super(Conv2dTest, self).__init__()
        from mxnet.gluon import nn
        with self.name_scope():
            self.conv1 = nn.Conv2D(filters, kernel_size=(kernel_h, kernel_w), strides=(stride_h, stride_w), padding=(padding_h, padding_w), use_bias=use_bias)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        return x

def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test convolution:')

    for kernel_h in [1, 2, 3, 5]:
        for stride_h in [1, 2, 3, 5]:
            for filters in [1, 3, 8, 16]:
                for padding_h in [0, 1, 2, 3]:
                    for use_bias in [True, False]:
                        # Create stride_h
                        net = Conv2dTest(
                            kernel_h=kernel_h, kernel_w=kernel_h,
                            stride_h=stride_h, stride_w=stride_h,
                            filters=filters, use_bias=use_bias,
                            padding_h=padding_h, padding_w=padding_h)
                        
                        # Make sure it's hybrid and initialized
                        net.hybridize()
                        net.collect_params().initialize()

                        pytorch_model = gluon2pytorch(net, dst_dir=None, pytorch_module_name='Conv2dTest')

                        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

                        gluon_output = net(mx.nd.array(input_np))
                        pytorch_output = pytorch_model(torch.FloatTensor(input_np))
                        check_error(gluon_output, pytorch_output)