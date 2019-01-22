It's the convertor of Gluon graph to a Pytorch model file + weights.

Firstly, we need to load (or create) Gluon Hybrid model:

```

class ReLUTest(mx.gluon.nn.HybridSequential):
    def __init__(self):
        super(ReLUTest, self).__init__()
        from mxnet.gluon import nn
        with self.name_scope():
            self.conv1 = nn.Conv2D(3, 32)
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = F.relu(self.relu(self.conv1(x)))
        return x


if __name__ == '__main__':
    net = ReLUTest()
    
    # Make sure it's hybrid and initialized
    net.hybridize()
    net.collect_params().initialize()

```

The next step - call the converter:

```
    pytorch_model = gluon2pytorch(net, [(1, 3, 224, 224)], dst_dir=None, pytorch_module_name='ReLUTest')
```

Finally, we can check the difference

```
    input_np = np.random.uniform(-1, 1, (1, 3, 224, 224))

    gluon_output = net(mx.nd.array(input_np))
    pytorch_output = pytorch_model(torch.FloatTensor(input_np))
    check_error(gluon_output, pytorch_output)
```