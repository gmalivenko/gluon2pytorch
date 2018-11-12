import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class UpsampleBilinearTest(mx.gluon.nn.HybridSequential):
    def __init__(self, h, w):
        super(UpsampleBilinearTest, self).__init__()
        from mxnet.gluon import nn
        self.h = h
        self.w = w

    def hybrid_forward(self, F, x):
        return  F.contrib.BilinearResize2D(x, height=self.h, width=self.w)


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test upsample addition:')

    for h in [16, 24, 32, 128]:
        for w in [16, 24, 32, 128]:
            net = UpsampleBilinearTest(h, w)

            # Make sure it's hybrid and initialized
            net.hybridize()
            net.collect_params().initialize()

            pytorch_model = gluon2pytorch(net, [(1, 3, 16, 16)], dst_dir=None, pytorch_module_name='UpsampleBilinearTest')

            input_np = np.random.uniform(0, 1, (1, 3, 16, 16))

            gluon_output = net(mx.nd.array(input_np))
            pytorch_output = pytorch_model(torch.FloatTensor(input_np))
            check_error(gluon_output, pytorch_output)
