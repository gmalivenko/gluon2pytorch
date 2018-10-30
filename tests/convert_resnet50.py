import argparse
import torch
import mxnet as mx
import numpy as np
import sys
from gluon2pytorch import gluon2pytorch


def check_error(gluon_output, pytorch_output, epsilon=1e-5):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test resnet50_v2:')

    from mxnet.gluon.model_zoo import vision as models
    net = models.resnet50_v2(classes=20)
    
    # Make sure it's hybrid and initialized
    net.hybridize()
    net.collect_params().initialize()

    pytorch_model = gluon2pytorch(net, dst_dir=None, pytorch_module_name='resnet50_v2')
    pytorch_model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    gluon_output = net(mx.nd.array(input_np))
    pytorch_output = pytorch_model(torch.FloatTensor(input_np))
    check_error(gluon_output, pytorch_output)