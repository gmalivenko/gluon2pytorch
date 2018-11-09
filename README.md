# gluon2pytorch

[![Build Status](https://travis-ci.com/nerox8664/gluon2pytorch.svg?branch=master)](https://travis-ci.com/nerox8664/gluon2pytorch)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/nerox8664/gluon2pytorch)

Gluon to PyTorch model convertor with script generation.

## Installation

```
git clone https://github.com/nerox8664/gluon2pytorch
cd gluon2pytorch
pip install -e . 
```

or you can use `pip`:

```
pip install gluon2pytorch
```

## How to use

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

## Supported layers

Layers:

* Linear
* Conv2d
* ConvTranspose2d (Deconvolution)
* MaxPool2d
* AvgPool2d
* Global average pooling (as special case of AdaptiveAvgPool2d)
* BatchNorm2d* 
* Padding2d (constant, reflection, replication)

Reshape:

* Flatten

Activations:

* ReLU
* LeakyReLU
* Sigmoid
* Softmax
* SELU

Element-wise:

* Addition
* Concatenation
* Subtraction
* Multiplication

Misc:

* clamp

## Classification models converted with gluon2pytorch


| Model | Top1 | Top5 | Params | FLOPs | Source weights | Remarks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-10 | 37.09 | 15.55 | 5,418,792 | 892.62M | osmr's repo | Success |
| ResNet-12 | 35.86 | 14.46 | 5,492,776 | 1,124.23M | osmr's repo | Success |
| ResNet-14 | 32.85 | 12.41 | 5,788,200 | 1,355.64M | osmr's repo | Success |
| ResNet-16 | 30.68 | 11.10 | 6,968,872 | 1,586.95M | osmr's repo | Success |
| ResNet-18 x0.25 | 49.16 | 24.45 | 831,096 | 136.64M | osmr's repo | Success |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 485.22M | osmr's repo | Success |
| ResNet-18 x0.75 | 33.25 | 12.54 | 6,675,352 | 1,045.75M | osmr's repo | Success |
| ResNet-18 | 29.13 | 9.94 | 11,689,512 | 1,818.21M | osmr's repo | Success |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,669.16M | osmr's repo | Success |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | 3,868.96M | osmr's repo | Success |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | 4,100.70M | osmr's repo | Success |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,586.30M | osmr's repo | Success |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | 7,818.04M | osmr's repo | Success |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | 11,304.85M | osmr's repo | Success |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | 11,536.58M | osmr's repo | Success |
| PreResNet-18 | 28.72 | 9.88 | 11,687,848 | 1,818.41M | osmr's repo | Success |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,669.36M | osmr's repo | Success |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,869.16M | osmr's repo | Success |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,100.90M | osmr's repo | Success |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | osmr's repo | Success |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,818.24M | osmr's repo | Success |
| PreResNet-152 | 20.70 | 5.32 | 60,185,256 | 11,305.05M | osmr's repo | Success |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,536.78M | Gluon Model Zoo| Success |
| PreResNet-200b | 21.10 | 5.64 | 64,666,280 | 15,040.27M | tornadomeet/ResNet | Success |
| ResNeXt-101 (32x4d) | 21.32 | 5.79 | 44,177,704 | 7,991.62M | Cadene's repo | Success |
| ResNeXt-101 (64x4d) | 20.60 | 5.41 | 83,455,272 | 15,491.88M | Cadene's repo | Success |
| SE-ResNet-50 | 22.51 | 6.44 | 28,088,024 | 3,877.01M | Cadene's repo | Success |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,600.01M | Cadene's repo | Success |
| SE-ResNet-152 | 21.48 | 5.77 | 66,821,848 | 11,324.62M | Cadene's repo | Success |
| SE-ResNeXt-50 (32x4d) | 21.06 | 5.58 | 27,559,896 | 4,253.33M | Cadene's repo | Success |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.00 | 48,955,416 | 8,005.33M | Cadene's repo | Success |
| SENet-154 | 18.84 | 4.65 | 115,088,984 | 20,742.40M | Cadene's repo | Success |
| DenseNet-121 | 25.11 | 7.80 | 7,978,856 | 2,852.39M | Gluon Model Zoo| Success |
| DenseNet-161 | 22.40 | 6.18 | 28,681,000 | 7,761.25M | Gluon Model Zoo| Success |
| DenseNet-169 | 23.89 | 6.89 | 14,149,480 | 3,381.48M | Gluon Model Zoo| Success |
| DenseNet-201 | 22.71 | 6.36 | 20,013,928 | 4,318.75M | Gluon Model Zoo| Success |
| DPN-68 | 23.57 | 7.00 | 12,611,602 | 2,338.71M | Cadene's repo | Success |
| DPN-98 | 20.23 | 5.28 | 61,570,728 | 11,702.80M | Cadene's repo | Success |
| DPN-131 | 20.03 | 5.22 | 79,254,504 | 16,056.22M | Cadene's repo | Success |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 496.34M | osmr's repo | Success |
| DarkNet Ref | 38.00 | 16.68 | 7,319,416 | 365.55M | osmr's repo | Success |
| SqueezeNet v1.0 | 40.97 | 18.96 | 1,248,424 | 828.30M | osmr's repo | Success |
| SqueezeNet v1.1 | 39.09 | 17.39 | 1,235,496 | 354.88M | osmr's repo | Success |
| SqueezeResNet v1.1 | 39.83 | 17.84 | 1,235,496 | 354.88M | osmr's repo | Success |
| ShuffleNetV2 x0.5 | 40.61 | 18.30 | 1,366,792 | 42.34M | osmr's repo | Success |
| ShuffleNetV2c x0.5 | 39.87 | 18.11 | 1,366,792 | 42.37M | tensorpack/tensorpack | Success |
| ShuffleNetV2 x1.0 | 33.76 | 13.22 | 2,278,604 | 147.92M | osmr's repo | Success |
| ShuffleNetV2c x1.0 | 30.74 | 11.38 | 2,279,760 | 148.85M | tensorpack/tensorpack | Success |
| ShuffleNetV2 x1.5 | 32.38 | 12.37 | 4,406,098 | 318.61M | osmr's repo | Success |
| ShuffleNetV2 x2.0 | 32.04 | 12.10 | 7,601,686 | 593.66M | osmr's repo | Success |
| 108-MENet-8x1 (g=3) | 43.62 | 20.30 | 654,516 | 40.64M | osmr's repo | Success |
| 128-MENet-8x1 (g=4) | 45.80 | 21.93 | 750,796 | 43.58M | clavichord93/MENet | Success |
| 228-MENet-12x1 (g=3) | 35.03 | 13.99 | 1,806,568 | 148.93M | clavichord93/MENet | Success |
| 256-MENet-12x1 (g=4) | 34.49 | 13.90 | 1,888,240 | 146.11M | clavichord93/MENet | Success |
| 348-MENet-12x1 (g=3) | 31.17 | 11.41 | 3,368,128 | 306.31M | clavichord93/MENet | Success |
| 352-MENet-12x1 (g=8) | 34.70 | 13.75 | 2,272,872 | 151.03M | clavichord93/MENet | Success |
| 456-MENet-24x1 (g=3) | 29.57 | 10.43 | 5,304,784 | 560.72M | clavichord93/MENet | Success |
| MobileNet x0.25 | 45.78 | 22.18 | 470,072 | 42.30M | osmr's repo | Success |
| MobileNet x0.5 | 36.12 | 14.81 | 1,331,592 | 152.04M | osmr's repo | Success |
| MobileNet x0.75 | 32.71 | 12.28 | 2,585,560 | 329.22M | Gluon Model Zoo| Success |
| MobileNet x1.0 | 29.25 | 10.03 | 4,231,976 | 573.83M | Gluon Model Zoo| Success |
| FD-MobileNet x0.25 | 56.19 | 31.38 | 383,160 | 12.44M | osmr's repo | Success |
| FD-MobileNet x0.5 | 42.62 | 19.69 | 993,928 | 40.93M | osmr's repo | Success |
| FD-MobileNet x1.0 | 35.95 | 14.72 | 2,901,288 | 146.08M | clavichord93/FD-MobileNet | Success |
| MobileNetV2 x0.25 | 48.89 | 25.24 | 1,516,392 | 32.22M | Gluon Model Zoo| Success |
| MobileNetV2 x0.5 | 35.51 | 14.64 | 1,964,736 | 95.62M | Gluon Model Zoo| Success |
| MobileNetV2 x0.75 | 30.82 | 11.26 | 2,627,592 | 191.61M | Gluon Model Zoo| Success |
| MobileNetV2 x1.0 | 28.51 | 9.90 | 3,504,960 | 320.19M | Gluon Model Zoo| Success |
| NASNet-A-Mobile | 25.37 | 7.95 | 5,289,978 | 587.29M | Cadene's repo | Success |
| InceptionV3 | 21.22 | 5.59 | 23,834,568 | 5,746.72M | Gluon Model Zoo| Success |


## Code snippets
Look at the `tests` directory.

## License
This software is covered by MIT License.