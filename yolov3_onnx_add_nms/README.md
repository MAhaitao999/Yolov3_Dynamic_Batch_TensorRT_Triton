### 环境
- onnx==1.6.0
onnx1.6.0对应的Opset11，是不支持UpSample操作的，为此在转成onnx时，将UpSample操作替换成了ReSize操作。

- TensorRT >= 7.2.1

由于TensorRT7.2之后开始支持[batchedNMSDynamicPlugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin)插件，因此我选取的TensorRT转换环境是
nvcr.io/nvidia/tensorrt:20.11-py3，对应的Triton Server部署环境是nvcr.io/nvidia/tritonserver:20.11-py3-clientsdk。同样的，20.12的转换和部署环境我也
进行了测试，也都没有问题。如果你选择低于20.11版本的话，那你就需要修改TensorRT的源码来自己注册这个插件了，网上也有相关的教程，在此不赘述。还需要注意的两点是：

- 某些GPU硬件可能转换会有问题：比如我在GTX 1660和Tesla P4两个GPU上测试了都没有问题，但是在M40上却出现了Coredump，具体原因未明，反正是在转NMS插件的时候发生的，
由于TensorRT7.2版本的报错太简略，未能定位到错误的位置。官方的issues上有说到TensorRT8将会对日志功能进行增强，到时候打算在TensorRT8上再调试调试，看看到底是什么原因。

- 20.11和20.12两个版本的镜像对Host上的NVIDIA驱动是有要求的，至少为[455.23](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)。

### 模型转换

#### 1. 将yolov3.weights和yolov3.cfg转成yolov3.onnx

#### 2. 给模型的输入输出重命名

```
```

### 模型部署


