import numpy as np
from PIL import Image, ImageDraw
from functools import partial
import os

import tritongrpcclient
import tritongrpcclient.model_config_pb2 as mc
import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


if __name__ == "__main__":
    
    #### prepare for image input
    input_image_path = "dog.jpg"
    input_resolution_yolov3_HW = (608, 608)
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    image_raw, image = preprocessor.process(input_image_path)
    shape_orig_WH = image_raw.size
    print(image.shape)
    triton_client = tritonhttpclient.InferenceServerClient("127.0.0.1:8000", verbose=False)
    inputs = []
    inputs.append(tritonhttpclient.InferInput("000_net", image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=True)
    outputs = []
    outputs.append(tritonhttpclient.InferRequestedOutput("082_convolutional", binary_data=True))
    outputs.append(tritonhttpclient.InferRequestedOutput("094_convolutional", binary_data=True))
    outputs.append(tritonhttpclient.InferRequestedOutput("106_convolutional", binary_data=True))
    import time
    iter_num = 1
    t1 = time.time()
    for i in range(iter_num):
        results = triton_client.infer("yolov3", inputs=inputs, outouts=outputs)
    t2 = time.time()
    print("inference cost {} ms".format(1000*(t2-t1)/iter_num))
    output1 = results.as_numpy("082_convolutional")
    output2 = results.as_numpy("094_convolutional")
    output3 = results.as_numpy("106_convolutional")
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    trtis_output = [output1, output2, output3]

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": input_resolution_yolov3_HW}


    postprocessor = PostprocessYOLO(**postprocessor_args)

    boxes, classes, scores = postprocessor.process(trtis_output, (shape_orig_WH))
    print(boxes)
    print(classes)
    print(scores)
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    output_image_path = 'output_dog.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
