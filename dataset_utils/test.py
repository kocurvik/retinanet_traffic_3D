import json
import pickle
import time
from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import os
import sys
import cv2

# Multithreded script to run the evaluation for the Transform2D
# and Transform3D methods. Online version displays the result.
# Offline version first saves all detections and then tracks
# them separately.

# Also includes a method to visually check the generated datasets.
from dataset_utils.utils import FolderVideoReader, deprocess_image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ))
    # import keras_retinanet.bin  # noqa: F401
    # __package__ = "keras_retinanet.bin"
    print(sys.path)


from dataset_utils.tracker import Tracker
from dataset_utils.warper import get_transform_matrix, get_transform_matrix_with_criterion
from dataset_utils.geometry import distance, computeCameraCalibration
from dataset_utils.writer import Writer
from keras_retinanet.utils.image import preprocess_image
from keras import backend as K

import keras_retinanet.models

TIMEOUT = 2000


def draw_raw_output(images, y_pred, threshold=0.5, cnt=None):
    images = deprocess_image(images)
    for i, image in enumerate(images):
        if cnt is not None:
            cv2.imwrite('frames/raw_t_{}_{}.jpg'.format(cnt, i), image)
        boxes = np.concatenate([y_pred[1][i, :, None], y_pred[0][i, :, :], y_pred[3][i, :, :]], 1)
        for box in boxes:
            if box[0] < threshold:
                continue
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            cy_1 = (1 - box[-1]) * (ymax - ymin) + ymin
            cy_0 = box[-1] * (ymax - ymin) + ymin

            cv2.line(image, (int(xmin), int(cy_0)), (int(xmax), int(cy_0)), (255, 0, 0), thickness=5)

            cv2.line(image, (int(xmin), int(ymin)), (int(xmin), int(ymax)), (0, 128, 0), thickness=5)
            cv2.line(image, (int(xmin), int(ymax)), (int(xmax), int(ymax)), (0, 128, 0), thickness=5)
            cv2.line(image, (int(xmax), int(ymax)), (int(xmax), int(ymin)), (0, 128, 0), thickness=5)
            cv2.line(image, (int(xmax), int(ymin)), (int(xmin), int(ymin)), (0, 128, 0), thickness=5)

            # cv2.line(image, (int(xmin), int(cy_1)), (int(xmax), int(cy_1)), (255, 0, 0), thickness=1)

        # if cnt is not None:
        #     cv2.imwrite('frames/raw_out_{}_{}.jpg'.format(cnt, i), image)
        cv2.imshow("Raw out", image)
        cv2.waitKey(1)


def test_video(model, video_path, json_path, im_w, im_h, batch, name, pair, out_path=None, compare=False, online=True, fake=False):
    with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp1, vp2, vp3, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

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
        out = cv2.VideoWriter(out_path, fourcc, 50.0, (frame.shape[1], frame.shape[0]))

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
            if online:
                draw_raw_output(images, y_pred, cnt = cnt)

    def postprocess():
        tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, pair = pair, threshold=0.3, compare=compare, fake=fake)
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
                cv2.imshow('frame', image_b)
                # cv2.imwrite('frames/detected_{}_{}.png'.format(counter, i), image_b)
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


def get_calib_params(im_h, im_w, json_path, pair, video_path):
    with open(json_path, 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']
    vp1, vp2, vp3, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])
    if os.path.exists(os.path.join(video_path, 'video_mask.png')):
        mask = cv2.imread(os.path.join(video_path, 'video_mask.png'), 0)
    else:
        mask = 255 * np.ones([1080, 1920], dtype=np.uint8)
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]
    if pair == '12':
        M, IM = get_transform_matrix_with_criterion(vp1, vp2, mask, im_w, im_h)
    elif pair == '23':
        M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h, vp_top=None)
    return IM, M, vp1, vp2, vp3


def track_detections(json_path, video_path, pair,  im_w, im_h, name, threshold, fake = False, write_name = None, keep = 5):
    print('Tracking: {} for t = {}'.format(name,threshold))

    IM, M, vp1, vp2, vp3 = get_calib_params(im_h, im_w, json_path, pair, video_path)

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold=threshold, pair = pair, fake=fake, write_name=write_name, keep = keep)
    tracker.read()


def export_detrac_detecions(json_path, video_path, pair,  im_w, im_h, name, fake = False, write_path=None):
    print('Exporting DETRAC detections for: {}'.format(name))

    IM, M, vp1, vp2, vp3 = get_calib_params(im_h, im_w, json_path, pair, video_path)

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, pair=pair, fake=fake)
    tracker.export_detrac_detections(write_path)


def test_dataset(images_path, ds_path, json_path, im_w, im_h, pair='23'):
    with open(ds_path, 'rb') as f:
        ds = pickle.load(f, encoding='latin-1', fix_imports=True)

    entry = ds[0]

    IM, M, vp1, vp2, vp3 = get_calib_params(im_h, im_w, json_path, pair, images_path)

    vp1_t = np.array([vp1], dtype="float32")
    vp1_t = np.array([vp1_t])
    vp1_t = cv2.perspectiveTransform(vp1_t, M)
    vp1_t = vp1_t[0][0]

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, 'none', threshold=0.5, pair = pair)

    pred_format = ['conf', 'x_min', 'y_min', 'x_max', 'y_max', 'centery']

    for entry in ds:
        frame = cv2.imread(os.path.join(images_path, entry['filename']))
        # frame = resize(frame,(1920,1080))
        frame = cv2.warpPerspective(frame, IM, (1920, 1080))
        print(frame.shape)

        # t_image = cv2.warpPerspective(frame, M, (480, 300), borderMode=cv2.BORDER_REPLICATE)

        boxes = entry['labels']
        boxes = [[1 if elem == 'conf' else box[elem] for elem in pred_format] for box in boxes]
        boxes = np.array(boxes)
        print(boxes)

        image_b = tracker.process(boxes, frame)
        # image_b = decode_3dbb(boxes, frame, IM, vp0, vp1, vp2, vp0_t)

        cv2.imshow('frame', image_b)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    vid_list = []
    calib_list = []

    if os.name == 'nt':
        vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
        results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'
    else:
        vid_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/dataset/'
        results_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/results/'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #
    # import tensorflow as tf
    # from keras import backend as k
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    #
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # k.tensorflow_backend.set_session(tf.Session(config=config))

    # for i in range(4, 7):
    #     # dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
    #     dir_list = ['session{}_right'.format(i)]
    #     vid_list.extend([os.path.join(vid_path, d, 'video.avi') for d in dir_list])
    #     calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])



    if os.name == 'nt':
        vid_path = 'D:/Skola/PhD/data/LuvizonDataset/dataset/'
        results_path = 'D:/Skola/PhD/data/LuvizonDataset/results/'
    else:
        vid_path = '/home/k/kocur15/data/luvizon/dataset/'
        results_path = '/home/k/kocur15/data/luvizon/results/'

    # sample vid_dict
    vid_dict = {1: [1], 2: [1], 3: [1], 4: [1], 5: [1]}

    for i in vid_dict.keys():
        vid_list.extend([os.path.join(vid_path, 'subset0{}'.format(i), 'video{:02d}'.format(j), 'video.h264') for j in vid_dict[i]])
        calib_list.extend([os.path.join(results_path, 'subset0{}'.format(i), 'video{:02d}'.format(j), 'calib.json') for j in vid_dict[i]])

    # if os.name == 'nt':
    #     vid_dir = 'D:/Skola/PhD/data/DETRAC/Insight-MVT_Annotation_Test/'
    # else:
    #     vid_dir = '/home/k/kocur15/data/DETRAC/Insight-MVT_Annotation_Test/'
    # vids = ['MVI_39031', 'MVI_39051','MVI_39211', 'MVI_39271', 'MVI_39371', 'MVI_39501', 'MVI_39511'] #, 'MVI_40742', 'MVI_40743', 'MVI_40863', 'MVI_40864']
    # # vids = ['MVI_39271'] #, 'MVI_40742', 'MVI_40743', 'MVI_40863', 'MVI_40864']
    # vid_list = [os.path.join(vid_dir, v) for v in vids]
    # calib_list = [os.path.join(vid_dir, v, 'calib.json') for v in vids]


    pair = '23'
    width = 640
    height = 360
    name = '{}_{}_{}_3'.format(width, height, pair)

    if os.name =='nt':
        model = keras_retinanet.models.load_model('D:/Skola/PhD/code/keras-retinanet/models/resnet50_{}_at30.h5'.format(name),
                                                  backbone_name='resnet50', convert=False)
    else:
        model = keras_retinanet.models.load_model('/home/k/kocur15/code/keras-retinanet/snapshots/{}/resnet50_{}_at30.h5'.format(name, name),
                                                  backbone_name='resnet50', convert=False)

    print(model.summary)
    model._make_predict_function()

    for vid, calib in zip(vid_list, calib_list):
        test_video(model, vid, calib, width, height, 4, name, pair, online=True, fake=False)# out_path='D:/Skola/PhD/code/keras-retinanet/video_results/center_6_12.avi')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    thresholds = [0.2]

    # name = '{}_{}_{}_1'.format(width, height, pair)
    # write_name = 'Transform3D_960_540_VP2VP3'
    #
    # for calib, vid in zip(calib_list, vid_list):
    #     track_detections(calib, vid, pair, width, height, name, 0.2, fake = False, keep=5, write_name=write_name)

    test_dataset('D:/Skola/PhD/data/Luvizon_boxed_23/images_0', 'D:/Skola/PhD/data/Luvizon_boxed_23/dataset_0.pkl',
                 'D:/Skola/PhD/data/LuvizonDataset/results/Set01/calib.json',
                 960, 540, pair='23')

    # if os.name == 'nt':
    #     write_path = 'D:/Skola/PhD/data/DETRAC/Transform3D'
    # else:
    #     write_path = '/home/k/kocur15/data/DETRAC/Transform3D'
    # name_list = [os.path.join(write_path, '{}_Det_Transform3D_640_360_23.txt'.format(v)) for v in vids]
    #
    #
    # for calib, vid, write_name in zip(calib_list, vid_list, name_list):
    #     export_detrac_detecions(calib, vid, pair, width, height, name, fake=False, write_path=write_name)

