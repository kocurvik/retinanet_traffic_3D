import math
import os
import sys
from copy import copy
import json

import cv2

import numpy as np
from dataset_utils.geometry import line, intersection
from dataset_utils.warper import warp_point

# Contains the Tracker class which does both online
# and offline tracking.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin  # noqa: F401
__package__ = "keras_retinanet.bin"

from ..utils.compute_overlap import compute_overlap

font = cv2.FONT_HERSHEY_SIMPLEX


class Tracker:
    def __init__(self, json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold = 0.7, pair='23', keep=5, compare = False, fake = False, write_name = None, save_often = True):
        self.detrac_detection_strings = []
        self.tracks = []
        self.assigned = []
        self.last_id = 0
        self.M = M
        self.IM = IM

        if pair == '23':
            self.vp1 = vp1
            self.vp2 = vp2
            self.vp3 = vp3
            self.vp1_t = warp_point(vp1, M)
        elif pair == '12':
            self.vp1 = vp3
            self.vp2 = vp2
            self.vp3 = vp1
            self.vp1_t = warp_point(vp3, M)

        print("VP1_T:{}".format(self.vp1_t))

        self.im_h = im_h
        self.im_w = im_w
        self.name = name
        self.threshold = threshold
        self.pair = pair
        self.keep = keep
        self.frame = 0
        self.save_often = save_often
        if write_name is None:
            self.write_name = self.name
        else:
            self.write_name = write_name
        self.write_path = os.path.join(os.path.dirname(json_path),'system_{}.json'.format(self.write_name, self.threshold*100))
        self.read_path = os.path.join(os.path.dirname(json_path),'detections_{}.json'.format(self.name))
        self.compare = compare
        self.fake = fake

        with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
            structure = json.load(file)
            if self.compare:
                self.dubska_cars = structure['cars']
            self.json_structure = {'cars':[], 'camera_calibration':structure['camera_calibration']}
        # self.json_structure = {'cars': []}


    def draw_box(self, box, id, image_b):
        bb_tt, center = self.get_bb(box)
        bb_tt = [tuple(point) for point in bb_tt]

        image_b = cv2.line(image_b, bb_tt[0], bb_tt[1], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[2], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[3], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[0], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[0], bb_tt[4], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[5], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[6], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[7], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[4], bb_tt[5], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[5], bb_tt[6], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[6], bb_tt[7], (0, 128, 0), 9)
        image_b = cv2.line(image_b, bb_tt[7], bb_tt[4], (0, 128, 0), 9)

        # image_b = cv2.putText(image_b, str(id), bb_tt[3], font, 4, (0, 0, 255), 2, cv2.LINE_AA)

        image_b = cv2.circle(image_b, (int(center[0]), int(center[1])), 5, (0, 255, 255), 5)

        return image_b, center

    def get_bb(self, box):
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        if self.fake:
            if (self.vp1_t[1] < ymin):
                cy_0 = ymin
            else:
                cy_0 = ymax
        else:
            cy_0 = box[-1] * (ymax - ymin) + ymin
        bb_t = []
        if (self.vp1_t[1] < ymin):
            if (xmin < self.vp1_t[0]) and (self.vp1_t[0] < xmax):
                # print("Case 1")
                cy = cy_0
                cx = xmin
                bb_t.append([cx, cy])
                bb_t.append([xmax, cy])
                bb_t.append([xmax, ymax])
                bb_t.append([cx, ymax])
                tx, ty = intersection(line([cx, cy], self.vp1_t), line([xmin, ymin], [xmin + 1, ymin]))
                bb_t.append([tx, ymin])
                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]

                center = (bb_tt[3] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp2)))

            elif self.vp1_t[0] < xmin:
                # print("Case 2")
                cx, cy = intersection(line([xmin, ymin], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, cy])
                bb_t.append([xmax, cy])
                bb_t.append([xmax, ymax])
                bb_t.append([cx, ymax])
                bb_t.append([xmin, ymin])

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[3] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp2)))
            else:  # vp1_t[0] > xmax
                # print("Case 3")
                cx, cy = intersection(line([xmax, ymin], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, cy])
                bb_t.append([cx, ymax])
                bb_t.append([xmin, ymax])
                bb_t.append([xmin, cy])
                bb_t.append([xmax, ymin])

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[1] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp3)))
        # elif (self.vp1_t[1] > ymax):
        else:
            if (xmin < self.vp1_t[0]) and (self.vp1_t[0] < xmax):
                # print("Case 4")
                cy = cy_0
                bb_t.append([xmin, cy])
                bb_t.append([xmax, cy])
                bb_t.append(intersection(line(self.vp1_t, [xmax, cy]), line([0, ymax], [1, ymax])))
                bb_t.append(intersection(line(self.vp1_t, [xmin, cy]), line([0, ymax], [1, ymax])))
                bb_t.append([xmin, ymin])
                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[2] + bb_tt[3]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp3), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp3), line(bb_tt[5], self.vp1)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp3), line(bb_tt[4], self.vp1)))

            elif self.vp1_t[0] < xmin:
                # print("Case 5")
                cx, cy = intersection(line([xmin, ymax], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, cy])
                bb_t.append([xmax, cy])
                bb_t.append(list(intersection(line([xmax, cy], self.vp1_t), line([xmin, ymax], [xmax, ymax]))))
                bb_t.append([xmin, ymax])

                bb_t.append([cx, ymin])
                bb_t.append([xmax, ymin])
                bb_t.append(
                    list(intersection(line(bb_t[2], [bb_t[2][0], bb_t[2][1] + 1]), line(self.vp1_t, [xmax, ymin]))))
                bb_t.append(list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line([xmin, 0], [xmin, 1]))))

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[3] + bb_tt[2]) / 2
            else:
                # print("Case 6: {}".format(self.vp1_t))
                print("xmax: {}, xmin: {}, ymin:{}, ymax:{}, c:{}".format(xmax, xmin, ymin, ymax, cy_0))
                cx, cy = intersection(line([xmax, ymax], self.vp1_t), line([0, cy_0], [1, cy_0]))

                bb_t.append([xmin, cy])
                bb_t.append([cx, cy])
                bb_t.append([xmax, ymax])
                bb_t.append(list(intersection(line([xmin, cy], self.vp1_t), line([xmin, ymax], [xmax, ymax]))))

                bb_t.append([xmin, ymin])
                bb_t.append([cx, ymin])
                bb_t.append(list(intersection(line([xmax, ymin], [xmax, ymax]), line(self.vp1_t, bb_t[5]))))
                bb_t.append(list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line(self.vp1_t, bb_t[4]))))

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[2] + bb_tt[3]) / 2
        # else:
            # if xmax < self.vp1_t[0]:
            #     if cy_0 < self.vp1_t[1]:
            #         cx, cy = intersection(line([xmin, ymin], [xmin + 1, ymin]), line([xmax, cy_0], self.vp1_t))
            #         bb_t.append([xmin, ymin])
            #         bb_t.append([cx, ymin])
            #         bb_t.append([cx, ymax])
            #         bb_t.append([xmin, ymax])
            #         bb_t.append(intersection(line(bb_t[0], self.vp1_t), line([xmax, cy_0] , [xmax + 1, cy_0])))
            #         bb_t.append(intersection(line(bb_t[1], self.vp1_t), line([xmax, cy_0] , [xmax + 1, cy_0])))
            #         bb_t.append(intersection(line(bb_t[2], self.vp1_t), line([xmax, ymax] , [xmax, ymax + 1])))
            #         bb_t.append(intersection(line(bb_t[3], self.vp1_t), line(bb_t[4] , [bb_t[4][0], bb_t[4][1] + 1])))
            #
            #         bb_t_array = np.array([[point] for point in bb_t], np.float32)
            #         bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
            #         bb_tt = [point[0] for point in bb_tt]
            #         center = (bb_tt[2] + bb_tt[3]) / 2
            #     else:
            #         cx, cy = intersection(line([xmin, ymax], [xmin + 1, ymax]), line([xmax, cy_0], self.vp1_t))
            #         bb_t.append([xmin, ymin])
            #         bb_t.append([cx, ymin])
            #         bb_t.append([cx, ymax])
            #         bb_t.append([xmin, ymax])
            #         tx, ty = intersection(line([cx, ymin], self.vp1_t), line([xmax, ymin], [xmax, ymin + 1]))
            #
            #         bb_t.append(intersection(line(bb_t[0], self.vp1_t), line([tx, ty] , [tx + 1, ty])))
            #         bb_t.append([tx, ty])
            #         bb_t.append(intersection(line(bb_t[2], self.vp1_t), line([xmax, ymax] , [xmax, ymax + 1])))
            #         bb_t.append(intersection(line(bb_t[3], self.vp1_t), line(bb_t[4] , [bb_t[4][0], bb_t[4][1] + 1])))
            #
            #         bb_t_array = np.array([[point] for point in bb_t], np.float32)
            #         bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
            #         bb_tt = [point[0] for point in bb_tt]
            #         center = (bb_tt[2] + bb_tt[3]) / 2
            # # elif xmin > self.vp1_t[0]:
            # #             #     ...
            # else:
        # bb_t = []
        # bb_t.append([xmin, ymin])
        # bb_t.append([xmax, ymin])
        # bb_t.append([xmax, ymax])
        # bb_t.append([xmin, ymax])
        # bb_t.append([xmin, ymin])
        # bb_t.append([xmax, ymin])
        # bb_t.append([xmax, ymax])
        # bb_t.append([xmin, ymax])
        #
        # bb_t_array = np.array([[point] for point in bb_t], np.float32)
        # bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
        # bb_tt = [point[0] for point in bb_tt]
        # center = (bb_tt[2] + bb_tt[3]) / 2

        return bb_tt, center

    def get_center(self, box):
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        if self.fake:
            if (self.vp1_t[1] < ymin):
                cy_0 = ymin
            else:
                cy_0 = ymax
        else:
            cy_0 = box[-1] * (ymax - ymin) + ymin

        bb_t = []
        if (self.vp1_t[1] < ymin):
            if (xmin < self.vp1_t[0]) and (self.vp1_t[0] < xmax):
                cy = cy_0
                cx = xmin
                bb_t.append([xmax, ymax])
                bb_t.append([cx, ymax])

            elif self.vp1_t[0] < xmin:
                cx, cy = intersection(line([xmin, ymin], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([xmax, ymax])
                bb_t.append([cx, ymax])
            else: # vp1_t[0] > xmax
                cx, cy = intersection(line([xmax, ymin], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, ymax])
                bb_t.append([xmin, ymax])
        else:
            if (xmin < self.vp1_t[0]) and (self.vp1_t[0] < xmax):
                cy = cy_0
                bb_t.append(intersection(line(self.vp1_t, [xmax, cy]), line([0, ymax], [1, ymax])))
                bb_t.append(intersection(line(self.vp1_t, [xmin, cy]), line([0, ymax], [1, ymax])))
            elif self.vp1_t[0] < xmin:
                cx, cy = intersection(line([xmin, ymax], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append(list(intersection(line([xmax, cy], self.vp1_t), line([xmin, ymax], [xmax, ymax]))))
                bb_t.append([xmin, ymax])
                if bb_t[0][0] < bb_t[1][0]:
                    return None
            else:
                cx, cy = intersection(line([xmax, ymax], self.vp1_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([xmax, ymax])
                bb_t.append(list(intersection(line([xmin, cy], self.vp1_t), line([xmin, ymax], [xmax, ymax]))))
                if bb_t[0][0] < bb_t[1][0]:
                    return None

        bb_t_array = np.array([[point] for point in bb_t], np.float32)
        bb_tt = cv2.perspectiveTransform(bb_t_array, self.IM)
        bb_tt = [point[0] for point in bb_tt]
        center = (bb_tt[0] + bb_tt[1])/2
        return center

    def process(self, boxes, image):
        image_b = copy(image)
        self.frame += 1
        if self.frame % 1000 == 0 and self.save_often:
            self.write()

        for box in boxes:
            if box[0] < self.threshold:
                continue
            track = self.get_track(box)
            image_b, center = self.draw_box(box,track.id,image_b)
            track.assign_center(center)
        self.remove()

        if self.compare:
            image_b = self.dubskaPoint(image_b)
        return image_b

    def process_offline(self, boxes):
        self.frame += 1
        if self.frame % 5000 == 0:
            self.write()

        for box in boxes:
            if box[0] < self.threshold:
                continue

            center = self.get_center(box)
            if center is not None:
                track = self.get_track(box)
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

    def process_detrac_detection(self, boxes):
        self.frame += 1

        for i, box in enumerate(boxes):
            bb_t, center = self.get_bb(box)
            bb_t = np.array(bb_t)
            x_min = np.min(bb_t[:, 0])
            x_max = np.max(bb_t[:, 0])
            y_min = np.min(bb_t[:, 1])
            y_max = np.max(bb_t[:, 1])
            t = box[0]
            s = '{}, {}, {}, {}, {}, {}, {}'.format(self.frame, i + 1, x_min, y_min, x_max - x_min, y_max - y_min, t)
            self.detrac_detection_strings.append(s)
        return

    def export_detrac_detections(self, out_path):
        with open(self.read_path, 'r') as file:
            list_of_boxes = json.load(file)
        for boxes in list_of_boxes:
            if len(boxes) == 0:
                self.process_detrac_detection(boxes)
            else:
                self.process_detrac_detection(np.array(boxes))

        with open(out_path, 'w') as f:
            for s in self.detrac_detection_strings:
                f.write('{}\n'.format(s))

    def remove(self):
        for i in reversed([i for (i, t) in enumerate(self.tracks) if t.check_misses(self.keep)]):
            self.addRecord(self.tracks[i])
            del self.tracks[i]

    def get_track(self, box):
        if len(self.tracks) == 0:
            self.last_id += 1
            new_track = self.Track(box, self.last_id, self.frame)
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
            # if center[0] > 1920 - 10 or center[0] < 10:
            #     continue
            # if center[1] > 1080 - 10 or center[1] < 10:
            #     continue

            # xmin = box[-5]
            # ymin = box[-4]
            # xmax = box[-3]
            # ymax = box[-2]

            # if ymax > self.im_h - 2.5:
            #     continue
            # if ymin < 2.5:
            #     continue
            # if xmax > self.im_w - 2.5:
            #     continue
            # if xmin < 2.5:
            #     continue

            posX.append(float(center[0]))
            posY.append(float(center[1]))
            frames.append(frame)

        if len(frames) < 5:
            return

        dist = math.sqrt(math.pow(posX[0] - posX[-1],2) + math.pow(posY[0] - posY[-1],2))
        if dist > 30:
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
            last_box = self.boxes[-1][np.newaxis, 1:5].astype(np.float64)
            query_box = box[np.newaxis,1:5].astype(np.float64)

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

    def dubskaPoint(self, image_b):
        for car in self.dubska_cars:
            try:
                idx = car["frames"].index(self.frame)
                posX = car["posX"][idx]
                posY = car["posY"][idx]
                image_b = cv2.circle(image_b, (int(posX), int(posY)), 5, (255,0,0), 3)
            except:
                pass
        return image_b

