import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch


class LinearTest(mx.gluon.nn.HybridSequential):
    def __init__(self, inputs=64, outputs=3, use_bias=False):
        super(LinearTest, self).__init__()
        from mxnet.gluon import nn
        with self.name_scope():
            self.linear1 = nn.Dense(
                units=outputs,
                in_units=inputs,
                use_bias=use_bias
            )

    def hybrid_forward(self, F, x):
        x = self.linear1(x)
        return x


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test linear:')

    for inputs in [1, 2, 16, 32, 128]:
        for outputs in [1, 2, 16, 32, 128]:
            for use_bias in [True, False]:
                # Create stride_h
                net = LinearTest(inputs=inputs, outputs=outputs, use_bias=use_bias)

                # Make sure it's hybrid and initialized
                net.hybridize()
                net.collect_params().initialize()

                pytorch_model = gluon2pytorch(net, [(1, inputs)], dst_dir=None, pytorch_module_name='LinearTest')

                input_np = np.random.uniform(0, 1, (1, inputs))

                gluon_output = net(mx.nd.array(input_np))
                pytorch_output = pytorch_model(torch.FloatTensor(input_np))
                check_error(gluon_output, pytorch_output)
