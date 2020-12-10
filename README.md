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

[TODO] 打算有时间加上Yolov4.

如果对你有帮助记得给我点个**Star**.

