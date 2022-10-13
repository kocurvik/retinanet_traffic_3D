import json

import cv2
import numpy as np

from dataset_utils.geometry import computeCameraCalibration, getWorldCoordinagesOnRoadPlane


def manual_scale(vid_path, calib_path, real_dist):
    with open(calib_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])

    projector = lambda x : getWorldCoordinagesOnRoadPlane(x, focal, roadPlane, pp)

    cap = cv2.VideoCapture(vid_path)
    ret, img = cap.read()


    clicked = []

    def mouse_callback(event, x, y, flags, params):
        if event == 1:
            clicked.append([x, y])
            print("Clicked x: {}, y: {}".format(x, y))

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('frame', mouse_callback)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    P1 = projector(np.array([clicked[0][0], clicked[0][1], 1]))
    P2 = projector(np.array([clicked[1][0], clicked[1][1], 1]))

    orig_dist = np.linalg.norm(P1 - P2)

    scale = real_dist / orig_dist
    structure['camera_calibration']['scale'] = scale

    with open(calib_path, 'w') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        json.dump(structure, file)


if __name__ == '__main__':
    vid_path = 'D:/Research/data/BASpeed/Zochova/video.m4v'
    calib_path = 'D:/Research/data/BASpeed/Zochova/calib.json'

    manual_scale(vid_path, calib_path, 0.6)