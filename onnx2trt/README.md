## 1. 对由darknet转成的静态onnx进行修改

打开Python Shell, 分别输入如下命令:

```py
> import onnx_graphsurgeon as gs
> import onnx
> graph1 = gs.import_onnx(onnx.load("../darknet2onnx/yolov3.onnx"))
> tensors = graph1.tensors()
> tensors["000_net"].shape[0] = gs.Tensor.DYNAMIC
> onnx.save(gs.export_onnx(graph1.cleanup().toposort()), "yolov3_dynamic.onnx")
```

上述步骤非常重要, 如果不进行上述操作, 由于onnx的batch size是固定的, 因此后续步骤生成的TensorRT模型就不具有动态性.

关于`onnx_graphsurgeon`的使用大家可以参考[这篇文章](https://elinux.org/TensorRT/ONNX).

## 2. trtexec 工具一阵梭哈

```sh
trtexec --onnx=yolov3_dynamic.onnx --explicitBatch --optShapes=000_net:16x3x608x608 --maxShapes=000_net:32x3x608x608 --minShapes=000_net:1x3x608x608 --shapes=000_net:16x3x608x608 --saveEngine=yolov3_dynamic.engine
```

经过漫长(大概十分钟吧)的等待, 你终于看到下面一行字:

```
&&&& PASSED TensorRT.trtexec
```

恭喜你转成功了!
