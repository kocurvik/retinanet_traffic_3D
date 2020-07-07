import os
import cv2
import keras_retinanet
import numpy as np
from dataset_utils.test import draw_raw_output
from dataset_utils.tracker import Tracker

from dataset_utils.warper import get_transform_matrix, get_transform_matrix_with_criterion
from dataset_utils.geometry import distance, computeCameraCalibration
from keras_retinanet.utils.image import preprocess_image


def test_video(vid_path, model, im_w=640, im_h=360):
    vp1 = np.array([660.0, -807.6190476190476])
    # vp2 = np.array([-258120.0, 30720.0])
    # vp2 = np.array([-255960.0, 34560.0])
    vp2 = np.array([9470.76923076923, 147.69230769230768])
    pp = np.array([960.5, 540.5])

    mask = 255 * np.ones([1080, 1920], dtype=np.uint8)

    vp1, vp2, vp3, _, _, focal = computeCameraCalibration(vp1, vp2, pp)

    print("Focal: {}".format(focal))

    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h)

    mg = np.array(np.meshgrid(range(im_w), range(im_h)))
    mg = np.reshape(np.transpose(mg, (1, 2, 0)), (im_w * im_h, 2))
    mg = np.array([[point] for point in mg]).astype(np.float32)
    map = np.reshape(cv2.perspectiveTransform(mg, np.array(IM)), (im_h, im_w, 2))

    json_path = 'D:/Skola/PhD/data/LuvizonDataset/'

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name='luvizon', pair = '23', threshold=0.5)

    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()

    while ret:
        t_image = cv2.remap(frame, map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        t_image = preprocess_image(t_image)
        # cv2.imshow("T_img", t_image)
        # cv2.waitKey(0)

        y_pred = model.predict(t_image[np.newaxis, ...])
        # draw_raw_output([t_image], y_pred)

        boxes = np.concatenate([y_pred[1][0, :, None], y_pred[0][0, :, :], y_pred[3][0, :, :]], 1)
        image_b = tracker.process(boxes, frame)
        cv2.imshow("Detected", image_b)
        cv2.waitKey(1)


        ret, frame = cap.read()





if __name__ == "__main__":
    vid_dir = 'D:/Skola/PhD/data/LuvizonDataset'

    model = keras_retinanet.models.load_model('D:/Skola/PhD/code/keras-retinanet/models/resnet50_640_360_23_3_at30.h5',
                                              backbone_name='resnet50', convert=False)

    print(model.summary)
    model._make_predict_function()

    for i in range(1, 6):
        vid_path = os.path.join(vid_dir, 'Set0{}_video01.h264'.format(i))
        test_video(vid_path, model)