## 准备TRT模型推理文件

```sh
cp ../onnx2trt/yolov3_dynamic.engine /your/path/repo/yolov3/1/model.plan
```

## 启动Triton Server镜像

```sh
docker run --runtime=nvidia --network=host -it --rm -v /your/path/repo:/repo nvcr.io/nvidia/tritonserver:20.08-py3 bash
```

## 启动服务

```sh
/opt/tritonserver/bin/tritonserver --model-store=/repo/ --strict-model-config=false --log-verbose 1
```

- `strict-model-config=false`表示模型配置文件不需要自己指定, Triton Server会帮你自动生成一份.

当生成如下日志时, 恭喜你模型部署成功!

```
I1210 09:10:47.913567 63 logging.cc:52] Deserialize required 3056008 microseconds.
I1210 09:10:47.978177 63 autofill.cc:225] TensorRT autofill: OK: 
W1210 09:10:47.978259 63 autofill.cc:269] No profiles specified in the model config. Will be selecting profile index 0 to determine the max_batch_size for yolov3
I1210 09:10:47.978332 63 model_config_utils.cc:629] autofilled config: name: "yolov3"
platform: "tensorrt_plan"
max_batch_size: 32
input {
  name: "000_net"
  data_type: TYPE_FP32
  dims: 3
  dims: 608
  dims: 608
}
output {
  name: "082_convolutional"
  data_type: TYPE_FP32
  dims: 255
  dims: 19
  dims: 19
}
output {
  name: "094_convolutional"
  data_type: TYPE_FP32
  dims: 255
  dims: 38
  dims: 38
}
output {
  name: "106_convolutional"
  data_type: TYPE_FP32
  dims: 255
  dims: 76
  dims: 76
}
default_model_filename: "model.plan"
...
I1210 09:11:27.601809 63 model_repository_manager.cc:925] successfully loaded 'yolov3' version 1
I1210 09:11:27.601848 63 model_repository_manager.cc:680] TriggerNextAction() 'yolov3' version 1: 0
I1210 09:11:27.601862 63 model_repository_manager.cc:695] no next action, trigger OnComplete()
I1210 09:11:27.601996 63 model_repository_manager.cc:515] VersionStates() 'yolov3'
...
I1210 09:11:27.604614 63 grpc_server.cc:3897] Started GRPCInferenceService at 0.0.0.0:8001
I1210 09:11:27.605000 63 http_server.cc:2679] Started HTTPService at 0.0.0.0:8000
I1210 09:11:27.646266 63 http_server.cc:2698] Started Metrics Service at 0.0.0.0:8002
```

## 启动客户端进行测试

```sh
docker run --runtime=nvidia --network=host -it --rm -v /your/path/repo:/repo nvcr.io/nvidia/tritonserver:20.08-py3-clientsdk bash
```

进入`/workspace/install/bin`目录下执行:

```sh
./perf_client -m yolov3 -b 17
```

当输出如下结果时, 说明服务端模型进行了推理.

```
*** Measurement Settings ***
  Batch size: 17
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client: 
    Request count: 5
    Throughput: 17 infer/sec
    Avg latency: 969643 usec (standard deviation 26043 usec)
    p50 latency: 963543 usec
    p90 latency: 1014324 usec
    p95 latency: 1014324 usec
    p99 latency: 1014324 usec
    Avg HTTP time: 953630 usec (send/recv 239915 usec + response wait 713715 usec)
  Server: 
    Inference count: 102
    Execution count: 6
    Successful request count: 6
    Avg request latency: 695450 usec (overhead 4 usec + queue 134 usec + compute input 27501 usec + compute infer 548955 usec + compute output 118856 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 17 infer/sec, latency 969643 usec
```
