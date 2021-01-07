# Yolov3_Dynamic_Batch_TensorRT_Triton

将Yolov3模型转成TensorRT模型, 跟官方给出的样例不同的是, 采用该项目进行转换得到的TensorRT模型可以采用TensorRT的Python接口进行动态Batch推理,
同时还可以以动态Batch的形式部署在Triton Inference Serving上. 可以说打通了Yolov3从模型训练结果到工业化部署这一段所有环节.

## 1. 实验环境

* [TensorRT](docker pull nvcr.io/nvidia/tensorrt:20.08-py3): 7.1.3
* [Triton Server](docker pull nvcr.io/nvidia/tritonserver:20.08-py3): v2.2.0
* [Triton Client](docker pull nvcr.io/nvidia/tritonserver:20.08-py3-clientsdk): v2.2.0

所有的模型转换步骤均在TensorRT镜像中进行.

```sh
docker run --runtime=nvidia --network=host -it --name tensorrt7_laborary -v `pwd`/tensorrt_workspace:/tensorrt_workspace nvcr.io/nvidia/tensorrt:20.08-py3 bash
```

安装必要的Python环境:

```sh
git clone https://github.com/MAhaitao999/Yolov3_Dynamic_Batch_TensorRT_Triton.git
cd Yolov3_Dynamic_Batch_TensorRT_Triton/
pip3 install -r requirements.txt -i https://pypi.douban.com/simple
```

默认的tensorrt:20.08-py3(TensorRT 7.1.3)镜像中并没有安装`onnx-graphsurgeon`环境, 需要自己手动安装, 而且只能以whl的方式. onnx-graphsurgeon的轮子我在`TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`里面找到了, 为省去大家寻找的烦恼, 我特地将其放在whl/目录下.

```sh
pip3 install whl/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```

## 2. 模型转换

### 2.1 yolov3权重文件下载

```sh
cd weights/
python3 download.py
cd ..
```

### 2.2 darknet 转 onnx

```sh
cd darknet2onnx/
python3 yolov3_to_onnx.py
cd ..
```

### 2.3 onnx 转 trt

```sh
cd onnx2trt/
按照里面的README.md进行操作
cd ..
```

### 2.4 执行TensorRT batch推理的脚本

```sh
cd trt_dynamic_inference/
python3 batch_inference.py --engine=../onnx2trt/yolov3_dynamic.engine
cd ..
```

## 3. 模型部署

```
cd triton_server_deployment/
按照里面的README.md进行操作即可
```

## 4. 后处理及NMS支持

已支持将后处理逻辑及NMS全部集成到trt模型文件中去, 减少了采用numpy进行大量后处理及NMS所产生的耗时.

[TODO] 打算有时间加上Yolov4.

如果对你有帮助记得给我点个**Star**.

