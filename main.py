# Before running this script, run following commands:
#   git clone https://github.com/tensorflow/models
#   cp -r ./models/research/object_detection ./object_detection
#   protoc object_detection/protos/*.proto --python_out=.
#   pip install -r requirements.txt

import sys
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import numpy as np
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
import timeit


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)

    model_dir = pathlib.Path(model_dir) / 'saved_model'

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.inputs)
print(detection_model.output_dtypes)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'],
                                                                              output_dict['detection_boxes'],
                                                                              image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image, class_ids):
    image_np = image
    output_dict = run_inference_for_single_image(model, image_np)

    boxes = []
    classes = []
    scores = []

    for i, x in enumerate(output_dict['detection_classes']):
        if x in class_ids and output_dict['detection_scores'][i] > 0.5:
            classes.append(x)
            boxes.append(output_dict['detection_boxes'][i])
            scores.append(output_dict['detection_scores'][i])

    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)

    vis_util.visualize_boxes_and_labels_on_image_array(image_np, boxes, classes, scores, category_index,
                                                       instance_masks=output_dict.get('detection_masks_reframed', None),
                                                       use_normalized_coordinates=True, line_thickness=2)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    sys.stderr.write('ERROR: Camera is not opened.')
    sys.exit()

w = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cam.get(cv2.CAP_PROP_FPS)
fps = 3.5
delay = round(1000/fps)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

if not out.isOpened():
    print('File open failed!')
    cam.release()
    sys.exit()

while True:
    ret, frame = cam.read()

    if not ret:
        print('cannot read video')
        break

    start_t = timeit.default_timer()

    predict_image = show_inference(detection_model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), range(1, 91))

    terminate_t = timeit.default_timer()

    out.write(predict_image)
    cv2.imshow('test', predict_image)
    print('FPS: ', 1 / (terminate_t - start_t))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()
