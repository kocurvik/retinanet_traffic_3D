import json
import pickle
import time
from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import os
import sys
import cv2
from dataset_utils.simple_tracker import SimpleTracker

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    # import keras_retinanet.bin  # noqa: F401
    # __package__ = "keras_retinanet.bin"
    print(sys.path)

# Script used to determine the FPS of a deployed system

# This differs from test.py by removing the live display and drawing
# from the live version.

from dataset_utils.tracker import Tracker
from dataset_utils.warper import get_transform_matrix, get_transform_matrix_with_criterion
from dataset_utils.geometry import distance, computeCameraCalibration
from dataset_utils.writer import Writer
from keras_retinanet.utils.image import preprocess_image
from keras import backend as K

import keras_retinanet.models


def run_video(model, video_path, json_path, im_w, im_h, batch, name, pair, fake=False):
    with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp1, vp2, vp3, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    cap = cv2.VideoCapture(os.path.join(video_path, 'video.avi'))
    mask = cv2.imread(os.path.join(video_path, 'video_mask.png'), 0)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1564)
    # for _ in range(1500):
    #     ret, frame = cap.read()
    ret, frame = cap.read()

    if pair == '12':
        M, IM = get_transform_matrix_with_criterion(vp1, vp2, mask, im_w, im_h)

    elif pair == '23':
        M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h)

    mg = np.array(np.meshgrid(range(im_w), range(im_h)))
    mg = np.reshape(np.transpose(mg, (1, 2, 0)), (im_w * im_h, 2))
    mg = np.array([[point] for point in mg]).astype(np.float32)
    map = np.reshape(cv2.perspectiveTransform(mg, np.array(IM)), (im_h, im_w, 2))

    q_frames = Queue(10)
    q_images = Queue(10)
    q_predict = Queue(10)
    e_stop = Event()

    vid_name = os.path.basename(os.path.normpath(video_path))

    def read():
        count = 0
        while (cap.isOpened() and not e_stop.isSet() and count < 1000):
            # read_time = time.time()
            images = []
            frames = []
            for _ in range(batch):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                frames.append(frame)
                image = cv2.bitwise_and(frame, frame, mask=mask)
                # t_image = cv2.remap(image, map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                # t_image = preprocess_image(t_image)
                t_image = cv2.resize(image, (640, 360))
                t_image = preprocess_image(t_image)
                images.append(t_image)
            # print("Read FPS: {}".format(batch / (time.time() - read_time)))
            q_images.put(images)
            q_frames.put(frames)
            count += 1
        e_stop.set()

    def inference():
        while (not e_stop.isSet()):
            try:
                images = q_images.get(timeout=100)
            except Empty:
                break
            gpu_time = time.time()
            y_pred = model.predict_on_batch(np.array(images))
            q_predict.put(y_pred)
            # print("GPU FPS: {}".format(batch / (time.time() - gpu_time)))

    def postprocess():
        # tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, pair = pair, threshold=0.2, compare=False, fake= fake, save_often=False)
        tracker = SimpleTracker(json_path, im_w, im_h, name, threshold=0.2)
        counter = 0
        full_time = time.time()
        total_time = time.time()
        while not e_stop.isSet():
            try:
                y_pred = q_predict.get(timeout=100)
                frames = q_frames.get(timeout=100)
            except Empty:
                # tracker.write()
                break
            # post_time = time.time()
            for i in range(len(frames)):
                boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)
                # if not fake:
                #     boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :], y_pred[3][i, :, :]], 1)
                # else:
                #     boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)

                image_b = tracker.process(boxes, frames[i])
                counter += 1
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     e_stop.set()
            # break
            # print("Post FPS: {}".format(batch / (time.time() - post_time)))
            # print("Total FPS: {}".format(batch / (time.time() - total_time)))

            total_time = time.time()
        print("Total FPS: {}".format(batch / (1000 * (time.time() - full_time))))

    inferencer = Thread(target=inference)

    reader = Thread(target=read)
    postprocesser = Thread(target=postprocess)

    reader.start()
    inferencer.start()
    postprocesser.start()

    reader.join()
    inferencer.join()
    postprocesser.join()


if __name__ == "__main__":
    vid_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/dataset/'
    results_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/results/'

    vid = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/dataset/session4_center'
    calib = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/results/session4_center/system_SochorCVIU_Edgelets_BBScale_Reg.json'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # names = ['480_270_23_1', '640_360_23_3', '960_540_23_1','270_480_12_0','360_640_12_0','540_960_12_0']

    names = ['640_360_ablation_0']

    for name in names:
        width = int(name.split("_")[0])
        height = int(name.split("_")[1])
        # pair = name.split("_")[4]

        print("Running for model: {}".format(name))

        model = keras_retinanet.models.load_model(
            '/home/k/kocur15/code/keras-retinanet/snapshots/{}/resnet50_{}_at30.h5'.format(name, name),
            backbone_name='resnet50', convert=False)
        print(model.summary)
        model._make_predict_function()

        pair = '23'
        run_video(model, vid, calib, width, height, 32, name, pair)  # , fake = True)
