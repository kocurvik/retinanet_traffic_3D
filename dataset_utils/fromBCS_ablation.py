import json
import pickle
import time

import cv2

import os
import sys
import numpy as np

sys.path[0:0] = [os.path.join(sys.path[0], '../../Mask_RCNN')]
# sys.path[0:0] = ['/home/kocur/code/Mask_RCNN']

# print(sys.path)

import coco as coco
import model as modellib

import os

# from dataset_utils.warper import get_transform_matrix, intersection, line
if os.name == 'nt':
    from dataset_utils.warper import get_transform_matrix, intersection, line, computeCameraCalibration

    COCO_MODEL_PATH = os.path.join('D:/Skola/PhD/code/Mask_RCNN', "mask_rcnn_coco.h5")
else:
    from warper import get_transform_matrix, intersection, line, computeCameraCalibration

    COCO_MODEL_PATH = os.path.join('/home/kocur/code/Mask_RCNN', "mask_rcnn_coco.h5")
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# COCO_MODEL_PATH = os.path.join('D:\Skola\PhD\code\MASK_RCNN', "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5

class BCS_boxer(object):
    def __init__(self, model, vid_list, pkl_path, images_path, im_w, im_h, lastvid=0, lastpos=0, save_often=False, n=0):
        # self.vehicles = [3, 6, 8]
        self.model = model
        self.vehicles = [3]
        self.vid_list = vid_list
        self.vid = lastvid
        self.pos = lastpos
        self.pkl_path = pkl_path
        if not os.path.exists(os.path.dirname(self.pkl_path)):
            os.makedirs(os.path.dirname(self.pkl_path))
        self.images_path = images_path
        if not os.path.exists(os.path.dirname(self.images_path)):
            os.makedirs(os.path.dirname(self.images_path))
        self.im_w = im_w
        self.im_h = im_h
        self.save_often = save_often
        self.n = n
        if self.vid != 0 or self.pos != 0:
            with open(self.pkl_path, "rb") as f:
                self.entries = pickle.load(f, encoding='latin-1', fix_imports=True)
        else:
            self.entries = []

    def process(self):
        N = len(self.vid_list)
        for v in range(self.vid, N):
            self.process_video(self.vid_list[v])
            self.pos = 0
            self.vid += 1

    def id(self):
        return self.pos * 1000 + self.vid

    def filename(self):
        return "{:02d}_{:08d}.png".format(self.vid, self.pos)

    def blob_boxer(self, image, roi):
        _, image = cv2.threshold(np.array(200 * image), 127, 255, cv2.THRESH_BINARY)

        # x_min = roi[1]
        # x_max = roi[3]
        # y_min = roi[0]
        # y_max = roi[2]

        box = {'class_id': 1,
               'x_min': roi[1],
               'x_max': roi[3],
               'y_min': roi[0],
               'y_max': roi[2]}
        return box


    def process_video(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)

        ret, frame = cap.read()

        while ret:
            if self.n != 0:
                if self.pos % self.n != 0:
                    self.pos += 1
                    ret, frame = cap.read()
                    continue

            boxes = []
            # Capture frame-by-frame
            # t_image = cv2.warpPerspective(frame, M, (self.im_w, self.im_h), borderMode=cv2.BORDER_CONSTANT)
            t_image = cv2.resize(frame,(self.im_w, self.im_h))

            # cv2.imshow('Warped',t_image)
            # cv2.waitKey(100)

            results = self.model.detect([t_image])

            r = results[0]


            for idx in range(len(r['class_ids'])):
                if r['class_ids'][idx] in self.vehicles:
                    box = self.blob_boxer(r['masks'][:, :, idx], r['rois'][idx])
                    boxes.append(box)

            entry = {'id': self.id(), 'filename': self.filename(), 'labels': boxes}

            self.entries.append(entry)

            targetpath = os.path.join(self.images_path, entry['filename'])
            if not os.path.exists(os.path.dirname(targetpath)):
                os.makedirs(os.path.dirname(targetpath))

            cv2.imwrite(targetpath, t_image)

            if self.save_often and self.pos % (1000*self.n) == 0:
                with open(self.pkl_path, "wb") as f:
                    pickle.dump(self.entries, f)
                    print("Saving, vid:{}, pos:{}".format(self.vid,self.pos))

            self.pos += 1
            ret, frame = cap.read()

        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.entries, f)
            print("Saving, vid:{}, pos:{}".format(self.vid,self.pos))

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
    # ds_path = 'D:/Skola/PhD/data/BCS_boxed/'

    vid_path = '/home/kocur/data/2016-ITS-BrnoCompSpeed/dataset/'
    ds_path = '/home/kocur/data/BCS_boxed/'

    vid_lists = []
    calib_lists = []
    for i in range(7):
        dir_list = []
        dir_list.append('session{}_center'.format(i))
        dir_list.append('session{}_left'.format(i))
        dir_list.append('session{}_right'.format(i))
        vid_list = [os.path.join(vid_path, d, 'video.avi') for d in dir_list]
        vid_lists.append(vid_list)

    pkl_paths = [os.path.join(ds_path, 'dataset_ablation_{}.pkl'.format(i)) for i in range(7)]
    image_paths = [os.path.join(ds_path, 'images_ablation_{}'.format(i)) for i in range(7)]

    # vid_list = [os.path.join(vid_path, d, 'video.avi') for d in dir_list]

    for i in range(4):
        boxer = BCS_boxer(model, vid_lists[i],  pkl_paths[i], image_paths[i], 960, 540, save_often=True, n = 25)
        boxer.process()
