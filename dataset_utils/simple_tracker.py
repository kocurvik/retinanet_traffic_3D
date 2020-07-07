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

from ..utils.compute_overlap import compute_overlap

font = cv2.FONT_HERSHEY_SIMPLEX


class SimpleTracker:
    def __init__(self, json_path, im_w, im_h, name, threshold = 0.7, keep=5):
        self.tracks = []
        self.assigned = []
        self.last_id = 0
        self.im_h = im_h
        self.scale_h = 1080 / im_h
        self.im_w = im_w
        self.scale_w = 1920 / im_w
        self.name = name
        self.threshold = threshold
        self.keep = keep
        self.frame = 0
        self.write_path = os.path.join(os.path.dirname(json_path),'system_Orig2D.json')
        self.read_path = os.path.join(os.path.dirname(json_path),'detections_{}.json'.format(self.name))


        with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
            structure = json.load(file)
            self.json_structure = {'cars':[], 'camera_calibration':structure['camera_calibration']}

    def draw_box(self, box, id, image_b):
        xmin = box[-4] * self.scale_w
        ymin = box[-3] * self.scale_h
        xmax = box[-2] * self.scale_w
        ymax = box[-1] * self.scale_h

        center = ((xmin + xmax)/2,ymax)
        image_b = cv2.rectangle(image_b,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0, 0, 255),2)
        image_b = cv2.putText(image_b, str(id), (int(xmin),int(ymax)), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
        image_b = cv2.circle(image_b, (int(center[0]), int(center[1])), 5, (0,255,0), 3)

        return image_b, center

    def get_center(self, box):
        xmin = box[-4] * self.scale_w
        ymin = box[-3] * self.scale_h
        xmax = box[-2] * self.scale_w
        ymax = box[-1] * self.scale_h
        center = ((xmin + xmax)/2,ymax)
        return center

    def process(self, boxes, image):
        image_b = copy(image)
        self.frame += 1
        if self.frame % 1000 == 0:
            self.write()


        for box in boxes:
            if box[0] < self.threshold:
                continue
            track = self.get_track(box)
            image_b, center = self.draw_box(box,track.id,image_b)
            track.assign_center(center)
        self.remove()

        return image_b

    def process_offline(self, boxes):
        self.frame += 1
        if self.frame % 5000 == 0:
            self.write()

        for box in boxes:
            if box[0] < self.threshold:
                continue
            track = self.get_track(box)
            center = self.get_center(box)
            track.assign_center(center)
        self.remove()

    def read(self):
        with open(self.read_path, 'r') as file:
            list_of_boxes = json.load(file)
        for boxes in list_of_boxes:
            if len(boxes) == 0:
                self.process_offline(boxes)
            else:
                self.process_offline(np.array(boxes))
        self.write()

    def remove(self):
        for i in reversed([i for (i, t) in enumerate(self.tracks) if t.check_misses(self.keep)]):
            self.addRecord(self.tracks[i])
            del self.tracks[i]

    def get_track(self, box):
        if len(self.tracks) == 0:
            self.last_id += 1
            new_track = self.Track(box,self.last_id, self.frame)
            self.tracks.append(new_track)
            return new_track

        max = 0.1
        max_track = None
        for track in self.tracks:
            if track.missing == -1:
                continue
            iou = track.iou(box)
            if max < iou:
                max = iou
                max_track = track
        if max_track is None:
            self.last_id += 1
            new_track = self.Track(box,self.last_id, self.frame)
            self.tracks.append(new_track)
            return new_track
        else:
            max_track.assign(box, self.frame)
            return max_track

    def addRecord(self, track):
        if len(track.frames) < 5:
            return

        frames = []
        posX = []
        posY = []

        for frame, center, box in zip(track.frames, track.centers, track.boxes):
            if center[0] > 1920 - 10 or center[0] < 10:
                continue
            if center[1] > 1080 - 10 or center[1] < 10:
                continue

            posX.append(float(center[0]))
            posY.append(float(center[1]))
            frames.append(frame)

        if len(frames) < 5:
            return

        dist = math.sqrt(math.pow(posX[0] - posX[-1],2) + math.pow(posY[0] - posY[-1],2))
        if dist > 100:
            entry = {'frames':track.frames, 'id': track.id, 'posX': posX, 'posY': posY}
            self.json_structure['cars'].append(entry)

    def write(self):
        with open(self.write_path,'w') as file:
            json.dump(self.json_structure, file)


    class Track:
        def __init__(self, box, id, frame):
            self.boxes = [box]
            self.frames = [frame]
            self.centers = []
            self.missing = -1
            self.id = id
            # self.vx = 0
            # self.vy = 0

        def iou(self, box):
            last_box = self.boxes[-1][np.newaxis, -4:].astype(np.float64)
            query_box = box[np.newaxis,-4:].astype(np.float64)

            return compute_overlap(last_box, query_box)

        def assign(self, box, frame):
            self.boxes.append(box)
            self.frames.append(frame)
            self.missing = -1

        def assign_center(self, center):
            self.centers.append(center)

        def check_misses(self, keep):
            self.missing += 1
            return self.missing > keep