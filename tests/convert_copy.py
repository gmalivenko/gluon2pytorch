import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class CopyTest(mx.gluon.nn.HybridSequential):
    def __init__(self):
        super(CopyTest, self).__init__()
        from mxnet.gluon import nn

    def hybrid_forward(self, F, x):
        return F.identity(x)


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test copy:')

    net = CopyTest()

    # Make sure it's hybrid and initialized
    net.hybridize()
    net.collect_params().initialize()

    pytorch_model = gluon2pytorch(net, [(1, 3, 224, 224)], dst_dir=None, pytorch_module_name='CopyTest')

    input_np = np.random.uniform(-1, 1, (1, 3, 224, 224))

    gluon_output = net(mx.nd.array(input_np))
    pytorch_output = pytorch_model(torch.FloatTensor(input_np))
    check_error(gluon_output, pytorch_output)
