import json
import os

import cv2
import numpy as np
from dataset_utils.geometry import computeCameraCalibration


def line_to_point(p1,p2,p3):
    return np.abs(np.cross(p2-p1,p3-p1,axis=2)/np.linalg.norm(p2-p1, axis=2))

def get_pts(vid_dir, json_path):
    video_path = os.path.join(vid_dir, 'video.avi')
    mask_path = os.path.join(vid_dir, 'video_mask.png')

    with open(json_path, 'r+') as file:
        # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
        structure = json.load(file)
        camera_calibration = structure['camera_calibration']

    vp0, vp1, vp2, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                      camera_calibration["pp"])
    vp0 = vp0[:-1] / vp0[-1]
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()


    frame = cv2.resize(frame, (640, 360))
    vp0 = vp0 / 3
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    accumulator = np.zeros([360,640])
    hsv[..., 1] = 255

    y,x = np.mgrid[0:360, 0:640]
    yx = np.stack([x,y], axis=2)


    cnt = 0
    while(cap.isOpened(), cnt < 10000):
        ret, frame2 = cap.read()
        frame2 = cv2.resize(frame2,(640,360))
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 7, 1.5, 0)

        d = line_to_point(yx, yx + flow, vp0)

        # for y in range(360):
        #     for x in range(640):
        #         p1 = np.array([x,y])
        #         d[y,x] = line_to_point(p1, p1 + flow[y,x], vp0)

        accepted = np.zeros_like(d)

        accepted[d < 3] = 1
        n = np.linalg.norm(flow, axis=2)
        accepted[n < 1] = 0
        accepted[flow[:,:,1]<0] = 0
        accumulator = accumulator + accepted

        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('frame2', bgr)

        cv2.imshow('accepted', accepted)
        cv2.imshow('frame', frame2)

        final = np.zeros_like(accumulator)
        final[accumulator > 0.01*np.max(accumulator)] = 1

        cv2.imshow('accumulator', accumulator/np.max(accumulator))
        cv2.imshow('norm', n/np.max(n))
        cv2.imshow('final', final)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    vid_dir = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset/session5_left'
    result_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/session5_left/system_SochorCVIU_Edgelets_BBScale_Reg.json'

    get_pts(vid_dir, result_path)
