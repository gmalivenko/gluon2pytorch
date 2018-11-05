import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class PadTest(mx.gluon.nn.HybridSequential):
    def __init__(self, pad_type, pad=0):
        super(PadTest, self).__init__()
        from mxnet.gluon import nn
        with self.name_scope():
            self.conv1 = nn.Conv2D(3, 32)
            self.relu = nn.Activation('relu')
        self.pad_type = pad_type
        self.pad = pad

    def hybrid_forward(self, F, x):
        x = F.pad(self.relu(self.conv1(x)), self.pad_type, (0, 0, 0, 0, self.pad, self.pad, self.pad, self.pad), constant_value=0)
        return x


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test pad:')

    for pad_type in ['reflect', 'edge', 'constant']:
        for pad in [0, 1, 2, 10, 20]:
            net = PadTest(pad_type=pad_type, pad=pad)

            # Make sure it's hybrid and initialized
            net.hybridize()
            net.collect_params().initialize()

            pytorch_model = gluon2pytorch(net, [(1, 3, 224, 224)], dst_dir=None, pytorch_module_name='PadTest')
            pytorch_model.eval()

            input_np = np.random.uniform(-1, 1, (1, 3, 224, 224))

            gluon_output = net(mx.nd.array(input_np))
            pytorch_output = pytorch_model(torch.FloatTensor(input_np))
            check_error(gluon_output, pytorch_output)
