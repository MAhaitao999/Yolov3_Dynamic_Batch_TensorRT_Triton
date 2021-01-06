### 环境
- onnx==1.6.0

onnx1.6.0对应的Opset11, 是不支持Upsample操作的, 为此在转成onnx时, 将UpSample操作替换成了Resize操作.

- TensorRT >= 7.2.1

由于TensorRT7.2之后开始支持[batchedNMSDynamicPlugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin)插件, 因此我选取的TensorRT转换环境是
nvcr.io/nvidia/tensorrt:20.11-py3, 对应的Triton Server部署环境是nvcr.io/nvidia/tritonserver:20.11-py3. 同样的, 20.12的转换和部署环境我也
进行了测试, 也都没有问题. 如果你选择低于20.11版本的话, 那你就需要修改TensorRT的源码来自己注册这个插件了, 网上也有相关的教程, 在此不赘述. 还需要注意的两点是:
    
- 某些GPU硬件可能转换会有问题: 比如我在GTX 1660和Tesla P4两个GPU上测试了都没有问题, 但是在M40上却出现了coredump, 具体原因未明, 反正是在转NMS插件的时候发生的,
由于TensorRT7.2版本的报错太简略, 未能定位到错误的位置. 官方的issues上有说到TensorRT8将会对日志功能进行增强, 到时候打算在TensorRT8上再调试调试, 看看到底是什么原因.
    
- 20.11和20.12两个版本的镜像对Host上的NVIDIA驱动是有要求的, 至少为[455.23](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).


### 模型转换

#### 1. 将yolov3.weights和yolov3.cfg转成yolov3.onnx

在`darknet2onnx/`文件夹下, 有个`yolov3_to_onnx_with_onnx1_6_0.py`文件, 执行该文件, 将生成的`yolov3.onnx`文件拷贝到当前目录下.

#### 2. 给模型的输入输出重命名

这一步的意义在于将输入输出重命名为标准成`input`, `outputx`这样比较标准的格式, 方便后面对onnx模型文件进行节点添加时代码的编写, 算是一个小trick吧.

```sh
python3 rename_ioname.py
```

这一步执行完成之后, 生成`yolov3_dynamic_rename.onnx`这个重命名之后的onnx模型文件.

#### 3. 给模型添加后处理

由于原始的yolov3模型输出的格式并不能直接满足batchedNMSDynamicPlugin插件的输入要求, 因此需要自己添加一些操作构造出满足插件输入要求的数据格式. 插件的输入有两个,
`Boxes input`和`Scores input`. 其中`Boxes input`的shape是`[batch_size, number_boxes, number_classes, number_box_parameters]`, `Scores input`的shape是
`[batch_size, number_boxes, number_classes]`, 在本例中对应的具体值分别为`[-1, 22743, 1, 4]`和`[-1, 8, 22743, 80]`, 注意`Boxes input`的`number_classes`这一项为
1, 这要求在设置插件的属性时将`shareLocation`属性设置成true, 表示`the boxes input are shared across all classes`. 添加后处理节点这一步代码写的很多, 很多numpy简单
一句话就能实现的操作需要用十几二十行onnx代码来替换, 比如numpy中简单的切片操作`[]`, 在onnx中你则需要将它转换成`Slice`op, `Slice`op不仅要准备输入输出tensor, 还要
准备`start`, `end`, `axis`这样的切片范围. 后处理部分的逻辑我是参考了`triton_server_deployment/clients`里面客户端进行后处理部分的代码.

```sh
python3 yolov3_add_postprocess.py
```

这一步执行完之后会生成`yolov3_dynamic_postprocess.onnx`添加完后处理之后的onnx模型文件.

#### 4. 给模型添加NMS节点

现在到了最关键的一步了, 给`yolov3_dynamic_postprocess.onnx`添加自定义的`BatchedNMSDynamic_TRT`op, 因为`BatchedNMSDynamic_TRT`这个名字的插件已经在TensorRT里面注册
过了, 所以下一步在转trt的时候TensorRT是能够识别出来的.

```sh
python3 onnx_add_nms_plugin.py
```

### 模型部署


