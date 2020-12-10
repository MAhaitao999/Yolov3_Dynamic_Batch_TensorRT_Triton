## 执行`batch_inference.py`

```sh
python3 batch_inference.py --engine=../onnx2trt/yolov3_dynamic.engine
```

看到如下输出:

```
Loaded engine: yolov3_dynamic.engine
Active Optimization Profile: 0
Engine/Binding Metadata
	Number of optimization profiles: 1
	Number of bindings per profile: 4
	First binding for profile 0: 0
	Last binding for profile 0: 3
Generating Random Inputs
	Using random seed: None
	Input [000_net] shape: (-1, 3, 608, 608)
	Profile Shapes for [000_net]: [kMIN (1, 3, 608, 608) | kOPT (16, 3, 608, 608) | kMAX (32, 3, 608, 608)]
	Input [000_net] shape was dynamic, setting inference shape to (17, 3, 608, 608)
cuda mem alloc cost: 29.211044311523438ms
Input Metadata
	Number of Inputs: 1
	Input Bindings for Profile 0: [0]
	Input names: ['000_net']
	Input shapes: [(17, 3, 608, 608)]
Output Metadata
	Number of Outputs: 3
	Output names: ['082_convolutional', '094_convolutional', '106_convolutional']
	Output shapes: [(17, 255, 19, 19), (17, 255, 38, 38), (17, 255, 76, 76)]
	Output Bindings for Profile 0: [1, 2, 3]
Inference Outputs: [[[[-1.34578317e-01  6.99492037e-01  4.87494841e-02 ...  3.48519906e-03
    -5.19855022e-01 -2.18663707e-01]
   [ 3.10347617e-01  1.62560776e-01  1.09931268e-02 ...  5.47992647e-01
     7.47665688e-02 -6.54516220e-01]
   [ 4.73091230e-02  3.29064846e-01 -7.57926702e-02 ...  3.33268225e-01
     7.53539652e-02 -5.69962263e-01]
   ...
   [-3.16027626e-02  3.14203620e-01 -1.18864678e-01 ...  3.52139771e-01
    -1.83140248e-01 -8.88761163e-01]
   [ 1.58334181e-01  2.85665154e-01 -1.59309909e-01 ...  3.52927744e-01
    -5.08402705e-01 -8.37002277e-01]
   [-2.41555810e-01  7.84065187e-01  8.85531306e-03 ... -1.36963725e-01
    -6.82207286e-01 -2.94923186e-01]]

  [[-1.06141880e-01  5.95748425e-01  2.35859543e-01 ...  4.04491454e-01
     9.89811778e-01  9.14608911e-02]
   [ 1.90860555e-01  9.37459990e-02  1.42472267e-01 ... -5.69525585e-02
    -4.59580868e-03 -1.07447878e-01]
   [ 1.54199556e-01  6.35828152e-02 -8.73239115e-02 ... -1.44323632e-01
    -9.02859122e-02 -9.59246010e-02]
   ...
```

恭喜你, 说明TensorRT的动态Batch是能够成功调用的.
