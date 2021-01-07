import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import os
import tritonclient.grpc
import tritonclient.grpc.model_config_pb2 as mc

import tritonclient.http
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException


def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_labels.txt')
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)

# Let's make sure that there are 80 classes, as expected for the COCO data set:
CATEGORY_NUM = len(ALL_CATEGORIES)
assert CATEGORY_NUM == 80


def draw_bboxes(image_raw, num_of_detections, boxes, scores, classes, bbox_color='green'):
    """Draw the bounding boxes on the original input image and return it.
    Keyword arguments:
    image_raw -- a raw PIL Image
    num_of_detections -- the number of object detected in the Image
    boxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    scores -- NumPy array containing the corresponding confidence for each object, with shape (N,)
    classes -- NumPy array containing the corresponding category for each object, with shape (N,)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'green')
    """
    draw = ImageDraw.Draw(image_raw)
    font = ImageFont.truetype("DejaVuSerif.ttf", 40)
    for i in range(num_of_detections):
        print("process {} objection".format(i))
        x1_coord, y1_coord, x2_coord, y2_coord = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        left = max(0, np.floor(x1_coord + 0.5).astype(int))
        top = max(0, np.floor(y1_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x2_coord + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y2_coord + 0.5).astype(int))
        print("the position of {} object is: {}, class is {}, confidence is {}".format(
            i, 
            ((left, top), (right, bottom)),
            ALL_CATEGORIES[int(classes[i])],
            scores[i]))
        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color, width=6)
        draw.text((left + 30, top + 15), '{0} {1:.2f}'.format(ALL_CATEGORIES[int(classes[i])], scores[i]), 
            fill=bbox_color, 
            font=font)

    return image_raw


image_name = "car.jpg"
image_raw = Image.open(image_name)
width, height = image_raw.size
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
origin_input_resolution = np.reshape(np.array([width, height, width, height], dtype=np.float32), (1, 1, 4))

inputs[1].set_data_from_numpy(origin_input_resolution, binary_data=True)
outputs = []
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_boxes", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_scores", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("nmsed_classes", binary_data=True))
outputs.append(tritonclient.http.InferRequestedOutput("num_detections", binary_data=True))
results = triton_client.infer("test", inputs=inputs, outputs=outputs)
num_detections = results.as_numpy("num_detections")
nmsed_boxes = results.as_numpy("nmsed_boxes")
nmsed_scores = results.as_numpy("nmsed_scores")
nmsed_classes = results.as_numpy("nmsed_classes")
# print("num_detections: ", num_detections.shape, num_detections)
# print("nmsed_scores: ", nmsed_scores.shape, nmsed_scores)
# print("nmsed_boxes: ", nmsed_boxes.shape, nmsed_boxes)
# print("nmsed_classes: ", nmsed_classes.shape, nmsed_classes)

batch_size = num_detections.shape[0]
print("batch_size is: ", batch_size)

print("Only Process the first image: ")
num_of_detections = num_detections[0][0]
print("The first image has detected {} objections".format(num_of_detections))

scores = nmsed_scores[0, 0:num_of_detections]
# print("scores is: ", scores)
boxes = nmsed_boxes[0, 0:num_of_detections, :]
# print("boxes is: ", boxes)
classes = nmsed_classes[0, 0:num_of_detections]
# print("classes is: ", classes)

draw_img = draw_bboxes(image_raw, num_of_detections, boxes, scores, classes)
output_image_path = 'output_' + image_name[:-4] + '.jpg'
draw_img.save(output_image_path, 'JPEG')
