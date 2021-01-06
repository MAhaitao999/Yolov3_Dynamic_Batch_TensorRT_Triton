import numpy as np
from PIL import Image, ImageDraw
from functools import partial
import os
import tritonclient.grpc
import tritonclient.grpc.model_config_pb2 as mc

import tritonclient.http
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

image_raw = Image.open("mug.jpg")
image_resized = image_raw.resize((608, 608), resample=Image.BICUBIC)
image_resized = np.array(image_resized, dtype=np.float32, order='C')
image_resized /= 255.0
image = np.transpose(image_resized, [2, 0, 1])
image = np.expand_dims(image, axis=0)
image = np.array(image, dtype=np.float32, order='C')
triton_client = tritonclient.http.InferenceServerClient("127.0.0.1:8000", verbose=False)
inputs = []
inputs.append(tritonclient.http.InferInput("input", image.shape, "FP32"))
inputs.append(tritonclient.http.InferInput("origin_input_resolution", (1, 1, 4), "FP32"))
inputs[0].set_data_from_numpy(image, binary_data=True)
origin_input_resolution = np.reshape(np.array([3361., 2521., 3361., 2521.], dtype=np.float32), (1, 1, 4))

inputs[1].set_data_from_numpy(origin_input_resolution, binary_data=True)
outputs = []
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_boxes", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_scores", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_classes", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("num_detections", binary_data=True))
results = triton_client.infer("yolov3_nms", inputs=inputs, outputs=outputs)
num_detections = results.as_numpy("num_detections")
nmsed_boxes = results.as_numpy("nmsed_boxes")
nmsed_scores = results.as_numpy("nmsed_scores")
nmsed_classes = results.as_numpy("nmsed_classes")
print("num_detections: ", num_detections)
print("nmsed_scores: ", nmsed_scores)
print("nmsed_boxes: ", nmsed_boxes)
print("nmsed_classes: ", nmsed_classes)

