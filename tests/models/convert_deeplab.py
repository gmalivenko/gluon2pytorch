import torch
import mxnet as mx
import numpy as np
import gluoncv
from gluon2pytorch import gluon2pytorch


def check_error(gluon_output, pytorch_output, epsilon=1e-4):
    if not isinstance(pytorch_output, tuple):
        pytorch_output = [pytorch_output]
        gluon_output = [gluon_output]

    for p, g in zip(pytorch_output, gluon_output):
        pytorch_output = p.data.numpy()
        gluon_output = g.asnumpy()

        error = np.max(pytorch_output - gluon_output)
        print('Error:', error)

        assert error < epsilon
        return error


if __name__ == '__main__':    
    net = gluoncv.model_zoo.DeepLabV3Plus(nclass=4, crop_size=224)
    net.hybridize()
    net.collect_params().initialize()

    pytorch_model = gluon2pytorch(net, [(1, 3, 224, 224)], dst_dir='../tmp/', pytorch_module_name='densenet169')
    pytorch_model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    gluon_output = net(mx.nd.array(input_np))[0]
    pytorch_output = pytorch_model(torch.FloatTensor(input_np))
    check_error(gluon_output, pytorch_output)


