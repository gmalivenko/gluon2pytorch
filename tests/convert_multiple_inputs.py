import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class Conv2dTest(mx.gluon.nn.HybridSequential):
    def __init__(self):
        super(Conv2dTest, self).__init__()
        from mxnet.gluon import nn
        with self.name_scope():
            self.conv1 = nn.Conv2D(32, 3)

    def hybrid_forward(self, F, x1, x2):
        x = self.conv1(x1) + self.conv1(x2)
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

    net = Conv2dTest()

    # Make sure it's hybrid and initialized
    net.hybridize()
    net.collect_params().initialize()

    pytorch_model = gluon2pytorch(net, [(1, 3, 224, 224), (1, 3, 224, 224)], dst_dir='../tmp', pytorch_module_name='Conv2dTest')

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    gluon_output = net(mx.nd.array(input_np), mx.nd.array(input_np))
    pytorch_output = pytorch_model(torch.FloatTensor(input_np), torch.FloatTensor(input_np))
    check_error(gluon_output, pytorch_output)
