import math
import os
import sys
from copy import copy
import json

import cv2

import numpy as np
from dataset_utils.geometry import line, intersection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin  # noqa: F401

__package__ = "keras_retinanet.bin"


class Writer:
    def __init__(self, json_path, name, threshold=0.1):
        self.list_of_boxes = []
        self.threshold = threshold
        self.frame = 0
        self.write_path = os.path.join(os.path.dirname(json_path), 'detections_{}.json'.format(name))

    def write(self):
        with open(self.write_path, 'w') as file:
            json.dump(self.list_of_boxes, file)

    def process(self, boxes):
        self.frame += 1
        if self.frame % 10000 == 0:
            self.write()

        last = 300

        for index, box in enumerate(boxes):
            if box[0] < self.threshold:
                last = index
                break
        self.list_of_boxes.append([box.tolist() for box in boxes[0:last]])
