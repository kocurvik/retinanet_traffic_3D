import json
import os
import sys
import time
from queue import Queue, Empty
from threading import Thread, Event

import tensorflow as tf

sys.path[0:0] = [os.path.join(sys.path[0], '../../Mask_RCNN')]
print(sys.path)
import coco as coco
import cv2
import numpy as np
import model as modellib
from dataset_utils.geometry import tangent_point_poly
from dataset_utils.tracker import Tracker
from dataset_utils.writer import Writer

from dataset_utils.warper import get_transform_matrix, get_transform_matrix_with_criterion
from dataset_utils.geometry import line, intersection, computeCameraCalibration


if os.name == 'nt':
    COCO_MODEL_PATH = os.path.join('D:/Skola/PhD/code/Mask_RCNN', "mask_rcnn_coco.h5")
else:
    COCO_MODEL_PATH = os.path.join('/home/k/kocur15/code/Mask_RCNN', "mask_rcnn_coco.h5")

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    DETECTION_MIN_CONFIDENCE = 0.5


def get_single_box_mask(image, M, vp, im_w, im_h):
    image = cv2.warpPerspective(np.array(200 * image), M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    if len(cnt) == 0:
        return None
    x_min, y_min, w, h = cv2.boundingRect(cnt)
    x_max = x_min + w
    y_max = y_min + h

    # x_min, y_min, w, h = cv2.boundingRect(cnt)
    # x_max = x_min + w
    # y_max = y_min + h

    if x_max < vp[0]:
        # box vlavo
        cls = 1
    elif x_min > vp[0]:
        # box vpravo
        cls = 3
    else:
        # box vstrede
        cls = 2

    hull = cv2.convexHull(cnt)
    V = [p[0].tolist() for p in hull]

    rt, lt = tangent_point_poly(vp, V, im_h)

    # image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    # image = cv2.line(image,tuple(rt),tuple(vp),(0,255,0))
    # image = cv2.line(image,tuple(lt),tuple(vp),(0,0,255))
    # image = cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(255,0,0),2)

    if cls == 1:
        cy1 = intersection(line([x_min, y_min], [x_min, y_max]), line(vp, lt))
        if vp[1] < 0:
            cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vp, rt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp, [x_max, y_min]))
        else:
            cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vp, rt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp, [x_max, y_max]))

    if cls == 3:
        cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vp, rt))
        if vp[1] < 0:
            cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vp, lt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp, [x_min, y_min]))
        else:
            cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vp, lt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp, [x_min, y_max]))
    if cls == 2:
        cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vp, rt))
        cy2 = intersection(line([x_min, y_min], [x_min, y_max]), line(vp, lt))

    # image = cv2.circle(image,tuple(cy1),2,(0,255,0))
    # image = cv2.circle(image,tuple(cy2),2,(0,0,255))
    # cv2.imshow("Detects", image)
    # cv2.waitKey(0)

    cy = min(cy1[1], cy2[1])
    centery = (cy - y_min) / (y_max - y_min)

    if centery < 0:
        centery = 0
    elif centery > 1:
        centery = 1

    # cv2.imshow("Debug", image)
    # cv2.waitKey(0)

    box = np.array([cls, x_min, y_min, x_max, y_max, centery])
    return box


def get_boxes_mask(y_pred, M, vp1_t, im_w, im_h):
    boxes = []
    for idx in range(len(y_pred['class_ids'])):
        if y_pred['class_ids'][idx] in [3]:
            box = get_single_box_mask(y_pred['masks'][:, :, idx].astype(np.uint8), M, vp1_t, im_w, im_h)
            if box is not None:
                boxes.append(box)
    return boxes


def test_video(model, video_path, json_path, im_w, im_h, batch, name, out_path=None, online=True):
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

    M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h)

    vp1_t = np.array([vp1], dtype="float32")
    vp1_t = np.array([vp1_t])
    vp1_t = cv2.perspectiveTransform(vp1_t, M)
    vp1_t = vp1_t[0][0]

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
                images.append(image)
            # print("Read FPS: {}".format(batch / (time.time() - read_time)))
            q_images.put(images)
            q_frames.put(frames)

    def read_offline():
        while (cap.isOpened() and not e_stop.isSet()):
            # read_time = time.time()
            images = []
            for _ in range(batch):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                image = cv2.bitwise_and(frame, frame, mask=mask)
                images.append(image)
            # print("Read FPS: {}".format(batch / (time.time() - read_time)))
            q_images.put(images)

    def inference():
        while (not e_stop.isSet()):
            try:
                images = q_images.get(timeout=100)
            except Empty:
                break
            gpu_time = time.time()

            # cv2.imshow('t_frame', images[0])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     e_stop.set()
            with graph.as_default():
                y_pred = model.detect(images)
            q_predict.put(y_pred)
            print("GPU FPS: {}".format(batch / (time.time() - gpu_time)))

    def postprocess():
        tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, pair = '23', threshold=0.2)
        counter = 0
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
                boxes = get_boxes_mask(y_pred[i], M, vp1_t, im_w, im_h)
                image_b = tracker.process(boxes, frames[i])
                if out_path is not None:
                    out.write(image_b)
                cv2.imshow('frame', image_b)
                counter += 1
                cv2.imwrite('frames/frame_{}_{}_{}.png'.format(vid_name, name, counter),image_b)
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
                boxes = get_boxes_mask(y_pred[i], M, vp1_t, im_w, im_h)
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


def track_detections(json_path, video_path, im_w, im_h, name, threshold):
    print('Tracking: {} for t = {}'.format(name,threshold))

    with open(json_path, 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp1, vp2, vp3, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])

    mask = cv2.imread(os.path.join(video_path, 'video_mask.png'), 0)

    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    frame = np.zeros([1080, 1920])
    M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h)

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold=threshold, pair = '23')
    tracker.read()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
    results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'

    # vid_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/dataset/'
    # results_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/results/'

    vid_list = []
    calib_list = []
    for i in range(4, 7):
        # if i == 5:
        #     dir_list = ['session{}_center'.format(i), 'session{}_right'.format(i)]
        # else:
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        # dir_list = ['session{}_right'.format(i), 'session{}_left'.format(i),]
        # dir_list = ['session{}_left'.format(i)]
        vid_list.extend([os.path.join(vid_path, d) for d in dir_list])
        calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])
        # calib_list.extend([os.path.join(results_path, d, 'system_dubska_optimal_calib.json') for d in dir_list])
        # calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_ManualCalib_ManualScale.json') for d in dir_list])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    name = 'mask_ablation'

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    global graph
    graph = tf.get_default_graph()

    # model._make_predict_function()

    for vid, calib in zip(vid_list, calib_list):
        test_video(model, vid, calib, 640, 360, 4, name, out_path=None, online = True)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    # thresholds = [0.2]

    for calib, vid in zip(calib_list, vid_list):
        for threshold in thresholds:
            track_detections(calib, vid, 360, 640, name, threshold)