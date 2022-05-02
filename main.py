# Before running this script, run following commands:
#   git clone https://github.com/tensorflow/models
#   cp -r ./models/research/object_detection ./object_detection
#   protoc object_detection/protos/*.proto --python_out=.
#   pip install -r requirements.txt

import sys
import timeit

import cv2
import numpy as np

from lib.video import BufferlessVideoCapture
from lib.model import load_model
from lib.inference import get_inference_image


def main():
    # model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

    model_name = 'tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8'
    detection_model = load_model(model_name)

    # print(detection_model.inputs)
    # print(detection_model.output_dtypes)

    cam_url = 'rtsp://192.168.1.32:554/stream2'
    capture = BufferlessVideoCapture(cam_url)

    if not capture.cap.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    w = round(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 3.5

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    if not out.isOpened():
        print('File open failed!')
        capture.cap.release()
        sys.exit()

    while True:
        start_t = timeit.default_timer()

        frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inference_image = get_inference_image(detection_model, frame, range(1, 91))
        out.write(inference_image)
        cv2.imshow('test', inference_image)

        terminate_t = timeit.default_timer()
        print('FPS: ', 1 / (terminate_t - start_t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
