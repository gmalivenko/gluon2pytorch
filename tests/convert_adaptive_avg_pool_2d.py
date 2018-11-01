import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class AdaptivePoolTest(mx.gluon.nn.HybridSequential):
    def __init__(self, pool_size=2):
        super(AdaptivePoolTest, self).__init__()
        from mxnet.gluon import nn
        self.pool_size = pool_size
        with self.name_scope():
            self.conv1 = nn.Conv2D(3, 32)

    def hybrid_forward(self, F, x):
        x = F.relu(F.contrib.AdaptiveAvgPooling2D((self.conv1(x)), output_size=self.pool_size))
        return x


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test adaptive_pool:')

    for pool_size in [1, 2, 3, 5, 7]:
        net = AdaptivePoolTest(pool_size=pool_size)

        # Make sure it's hybrid and initialized
        net.hybridize()
        net.collect_params().initialize()

        pytorch_model = gluon2pytorch(net, dst_dir=None, pytorch_module_name='AdaptivePoolTest')

        input_np = np.random.uniform(-1, 1, (1, 3, 224, 224))

        gluon_output = net(mx.nd.array(input_np))
        pytorch_output = pytorch_model(torch.FloatTensor(input_np))
        check_error(gluon_output, pytorch_output)
