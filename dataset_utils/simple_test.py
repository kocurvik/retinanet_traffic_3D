import json
import pickle
import time
from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import os
import sys
import cv2

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ))
    # import keras_retinanet.bin  # noqa: F401
    # __package__ = "keras_retinanet.bin"
    print(sys.path)


from dataset_utils.simple_tracker import SimpleTracker
from dataset_utils.warper import decode_3dbb, get_transform_matrix, computeCameraCalibration
from dataset_utils.writer import Writer
from keras_retinanet.utils.image import preprocess_image
from keras import backend as K

import keras_retinanet.models


def test_video(model, video_path, json_path, im_w, im_h, batch, name, out_path=None, online=True):
    # with open(json_path, 'r+') as file:
    #      with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:


    cap = cv2.VideoCapture(os.path.join(video_path, 'video.avi'))
    mask = cv2.imread(os.path.join(video_path, 'video_mask.png'), 0)

    ret, frame = cap.read()
    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(out_path, fourcc, 25.0, (frame.shape[1], frame.shape[0]))

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
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                frames.append(frame)
                image = cv2.bitwise_and(frame, frame, mask=mask)
                t_image = cv2.resize(image, (im_w,im_h))
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
        while (cap.isOpened() and not e_stop.isSet()):
            images = []
            for _ in range(batch):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                image = cv2.bitwise_and(frame, frame, mask=mask)
                t_image = cv2.resize(image, (im_w, im_h))
                t_image = preprocess_image(t_image)
                images.append(t_image)
            q_images.put(images)

    def inference():
        while (not e_stop.isSet()):
            try:
                images = q_images.get(timeout=100)
            except Empty:
                break
            gpu_time = time.time()
            y_pred = model.predict_on_batch(np.array(images))
            q_predict.put(y_pred)
            print("GPU FPS: {}".format(batch / (time.time() - gpu_time)))

    def postprocess():
        tracker = SimpleTracker(json_path, im_w, im_h, name, threshold=0.2)
        total_time = time.time()
        while not e_stop.isSet():
            try:
                y_pred = q_predict.get(timeout=100)
                frames = q_frames.get(timeout=100)
            except Empty:
                tracker.write()
                break
            # post_time = time.time()
            for i in range(len(frames)):
                boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)
                image_b = tracker.process(boxes, frames[i])
                # cv2.imwrite('frame_a_{}.png'.format(i), image_b)
                if out_path is not None:
                    out.write(image_b)
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
                y_pred = q_predict.get(timeout=100)
            except Empty:
                writer.write()
                break
            for i in range(y_pred[0].shape[0]):
                boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :]], 1)
                writer.process(boxes)
                frame_cnt += 1
            # print("Total FPS: {}".format(batch / (time.time() - total_time)))
            print("Video: {} at frame: {}, FPS: {}".format(vid_name, frame_cnt, frame_cnt / (time.time()-total_time)))
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
    cv2.destroyAllWindows()


def track_detections(json_path, im_w, im_h, name, threshold):
    tracker = SimpleTracker(json_path, im_w, im_h, name, threshold=threshold)
    tracker.read()

if __name__ == "__main__":

    vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
    results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'

    # vid_path = '/home/kocur/data/2016-ITS-BrnoCompSpeed/dataset/'
    # results_path = '/home/kocur/data/2016-ITS-BrnoCompSpeed/results/'

    vid_list = []
    calib_list = []
    for i in range(6, 7):
        # dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        # dir_list = ['session{}_left'.format(i), 'session{}_right'.format(i)]
        dir_list = ['session{}_left'.format(i)]
        vid_list.extend([os.path.join(vid_path, d) for d in dir_list])
        calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])


    model = keras_retinanet.models.load_model('D:/Skola/PhD/code/keras-retinanet/models/resnet50_ablation_640_360.h5',
                                              backbone_name='resnet50', convert=False)

    # model = keras_retinanet.models.load_model('/home/kocur/code/keras-retinanet/models/resnet50_ablation_640_360.h5',
    #                                           backbone_name='resnet50', convert=False)

    print(model.summary)
    model._make_predict_function()

    name = 'ablation_640_360'

    for vid, calib in zip(vid_list, calib_list):
        test_video(model, vid, calib, 640, 360, 16, name, online=True)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
    # # thresholds = [0.1]
    #
    for calib in calib_list:
        for threshold in thresholds:
            track_detections(calib, 640, 360, name, threshold)

