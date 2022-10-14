import argparse
import json
import math
import pickle
import time
from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import os
import sys
import cv2

# Multithreded script to run the evaluation for the Transform2D and Transform3D methods. Online version displays the
# result. Offline version first saves all detections and then tracks them separately.

# Also includes a method to visually check the generated datasets.
from dataset_utils.tracker import Tracker
from dataset_utils.utils import FolderVideoReader, deprocess_image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    # import keras_retinanet.bin  # noqa: F401
    # __package__ = "keras_retinanet.bin"
    print(sys.path)

from dataset_utils.warper import get_transform_matrix, get_transform_matrix_with_criterion, warp_point
from dataset_utils.geometry import distance, computeCameraCalibration, intersection, line, \
    getWorldCoordinagesOnRoadPlane
from dataset_utils.writer import Writer
from keras_retinanet.utils.image import preprocess_image
from keras import backend as K

import keras_retinanet.models

TIMEOUT = 20

font = cv2.FONT_HERSHEY_SIMPLEX


class TrackerBA(Tracker):
    def __init__(self, projector, fps, json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold=0.7, pair='23', keep=5,
                 compare=False, fake=False, write_name=None, save_often=True):
        super().__init__(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold, pair, keep, compare, fake, write_name, save_often)

        self.projector = projector
        self.fps = fps

    def draw_box_with_speed(self, track, box, image_b):
        bb_tt, center = self.get_bb(box)
        track.assign_center(center)

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

        id = track.id
        speed = track.get_speed(self.projector, self.fps)

        image_b = cv2.putText(image_b, '{}:{:.2f}'.format(id, speed), bb_tt[3], font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image_b = cv2.circle(image_b, (int(center[0]), int(center[1])), 5, (0, 255, 255), 5)

        return image_b, center

    def process(self, boxes, image):
        image_b = np.copy(image)
        self.frame += 1
        if self.frame % 1000 == 0 and self.save_often:
            self.write()

        for box in boxes:
            if box[0] < self.threshold:
                continue
            track = self.get_track(box)
            image_b, center = self.draw_box_with_speed(track, box, image_b)
        self.remove()
        return image_b

    class Track(Tracker.Track):
        def __init__(self, box, id, frame):
            super().__init__(box, id, frame)

        def get_speed(self, projector, fps):
            if len(self.centers) < 3:
                return 5

            centers_world_array = np.array([projector(np.array([p[0], p[1], 1])) for p in self.centers])
            dists = np.linalg.norm(centers_world_array[1:] - centers_world_array[:-1], axis=1)
            frame_diffs = np.array(self.frames)[1:] - np.array(self.frames)[:-1]
            speeds = 3.6 * dists / (frame_diffs / fps)

            return np.median(speeds)


def test_video(model, video_path, json_path, im_w, im_h, batch, name, pair, out_path=None, compare=False, online=True,
               fake=False, show=False):
    # This function runs the detections. If online=True then the system runs with tracking and outputting the resulting
    # video. If online=False then the detections are saved to a file and have to be tracked later. The fake parameter is
    # for ablation experiments where perspective transformation is used but only 2D bboxes are output (hence fake since
    # we fake the c_c param to be zero to allow for the use of the same code)

    with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    scale = camera_calibration['scale']
    projector = lambda x : scale * getWorldCoordinagesOnRoadPlane(x, focal, roadPlane, pp)

    if os.path.isdir(video_path):
        cap = FolderVideoReader(video_path)
        video_dir = video_path
    else:
        cap = cv2.VideoCapture(video_path)
        video_dir = os.path.dirname(video_path)
    if os.path.exists(os.path.join(video_dir, 'video_mask.png')):
        mask = cv2.imread(os.path.join(video_dir, 'video_mask.png'), 0)
    else:
        ret, img = cap.read()
        mask = 255 * np.ones(img.shape[:2], dtype=np.uint8)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1564)
    # for _ in range(1500):
    #     ret, frame = cap.read()
    ret, frame = cap.read()

    if pair == '12':
        M, IM = get_transform_matrix_with_criterion(vp1, vp2, mask, im_w, im_h, constraint=0.8)

    elif pair == '23':
        M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h, constraint=0.8, vp_top=None)

    mg = np.array(np.meshgrid(range(im_w), range(im_h)))
    mg = np.reshape(np.transpose(mg, (1, 2, 0)), (im_w * im_h, 2))
    mg = np.array([[point] for point in mg]).astype(np.float32)
    map = np.reshape(cv2.perspectiveTransform(mg, np.array(IM)), (im_h, im_w, 2))

    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    q_frames = Queue(10)
    q_images = Queue(10)
    q_predict = Queue(10)
    e_stop = Event()

    vid_name = os.path.basename(os.path.normpath(video_path))

    def read():
        while (cap.isOpened() and not e_stop.isSet()):
            # read_time = time.time()
            images = []
            frames = []
            for _ in range(batch):
                # for _ in range(50):
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    if len(images) > 0:
                        q_images.put(images)
                        q_frames.put(frames)
                    q_images.put(None)
                    q_frames.put(None)
                    break
                frames.append(frame)
                image = cv2.bitwise_and(frame, frame, mask=mask)
                # t_image = cv2.warpPerspective(image, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)
                t_image = cv2.remap(image, map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                # cv2.imshow('transform', t_image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     e_stop.set()
                # t_image = t_image[:, :, ::-1]
                t_image = preprocess_image(t_image)
                images.append(t_image)
            # print("Read FPS: {}".format(batch / (time.time() - read_time)))
            q_images.put(images)
            q_frames.put(frames)

    def read_offline():
        cnt = 0
        while (cap.isOpened() and not e_stop.isSet()):
            # read_time = time.time()
            images = []
            for _ in range(batch):
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    q_images.put(images)
                    q_images.put(None)
                    # e_stop.set()
                    break
                image = cv2.bitwise_and(frame, frame, mask=mask)
                t_image = cv2.remap(image, map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                t_image = preprocess_image(t_image)
                images.append(t_image)
            # print("Read FPS: {}".format(batch / (time.time() - read_time)))
            q_images.put(images)

    def inference():
        cnt = 0
        while (not e_stop.isSet()):
            try:
                images = q_images.get(timeout=TIMEOUT)
                if images is None:
                    break
            except Empty:
                break
            gpu_time = time.time()
            cnt += 1
            # cv2.imshow('t_frame', images[0])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     e_stop.set()
            y_pred = model.predict_on_batch(np.array(images))
            q_predict.put(y_pred)
            print("GPU FPS: {}".format(batch / (time.time() - gpu_time)))
            # if online:
            #     draw_raw_output(images, y_pred, cnt=cnt)

    def postprocess():
        tracker = TrackerBA(projector, 30.0, json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, pair=pair, threshold=0.3, compare=compare, fake=fake)
        counter = 0
        total_time = time.time()
        while not e_stop.isSet():
            try:
                y_pred = q_predict.get(timeout=TIMEOUT)
                frames = q_frames.get(timeout=TIMEOUT)
                if frames is None or y_pred is None:
                    tracker.write()
                    break
            except Empty:
                tracker.write()
                break
            # post_time = time.time()
            counter += 1
            for i in range(len(frames)):
                if not fake:
                    boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :], y_pred[3][i, :, :]], 1)
                else:
                    boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)

                # cv2.imwrite('frames/original_{}_{}.png'.format(counter, i), frames[i])
                image_b = tracker.process(boxes, frames[i])

                if out_path is not None:
                    out.write(image_b)
                if show:
                    cv2.imshow('frame', image_b)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        e_stop.set()

            # break
            # print("Post FPS: {}".format(batch / (time.time() - post_time)))
            # print("Total FPS: {}".format(batch / (time.time() - total_time)))
            # total_time = time.time()

    def postprocess_offline():
        writer = Writer(json_path, name)
        total_time = time.time()
        frame_cnt = 1
        while not e_stop.isSet():
            try:
                y_pred = q_predict.get(timeout=TIMEOUT)
            except Empty:
                writer.write()
                break
            for i in range(y_pred[0].shape[0]):
                if (len(y_pred) >= 4):
                    boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :], y_pred[3][i, :, :]], 1)
                else:
                    boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)
                writer.process(boxes)
                frame_cnt += 1
            # print("Total FPS: {}".format(batch / (time.time() - total_time)))
            print("Video: {} at frame: {}, FPS: {}".format(vid_name, frame_cnt, frame_cnt / (time.time() - total_time)))
            # total_time = time.time()

    inferencer = Thread(target=inference)

    if online:
        reader = Thread(target=read)
        postprocesser = Thread(target=postprocess)
    else:
        reader = Thread(target=read_offline)
        postprocesser = Thread(target=postprocess_offline)

    reader.start()
    inferencer.start()
    postprocesser.start()

    reader.join()
    inferencer.join()
    postprocesser.join()

    if out_path is not None:
        out.release()



def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', default=None, help='Path to output video')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for inference')
    parser.add_argument('-s', '--show', default=False, action='store_true', help='Whether to show video')
    parser.add_argument('model_path', help='Path to model')
    parser.add_argument('vid_path', help='Path to video')
    parser.add_argument('calib_path', help='Path to calib file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # model 640_360_23

    args = parse_command_line()

    model = keras_retinanet.models.load_model(args.model_path, backbone_name='resnet50', convert=False)

    print(model.summary)
    model._make_predict_function()

    vid_path = 'D:/Research/data/BASpeed/Zochova/video.m4v'
    calib_path = 'D:/Research/data/BASpeed/Zochova/calib.json'

    test_video(model, args.vid_path, args.calib_path, 640, 360, args.batch_size, 'result', '23', online=True, fake=False, out_path=args.output_path, show=args.show)