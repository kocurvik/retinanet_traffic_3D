import os
import cv2
import json
import sys
import numpy as np

from dataset_utils.utils import FolderVideoReader
from dataset_utils.diamond_accumulator import Accumulator

element_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
element_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
element_loc_max = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def get_vp1(cap, mask, debug=False):
    lk_params = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=20, qualityLevel=0.4, minDistance=10, blockSize=11)
    ret, prev_frame = cap.read()
    # prev_frame = cv2.bitwise_and(prev_frame, prev_frame, mask=mask)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    acc = Accumulator(size=256, debug=debug, height=mask.shape[0], width=mask.shape[1])
    cnt = 0
    while ret and cnt < 1000:
        ret, next_frame = cap.read()
        if not ret or next_frame is None:
            continue
        # next_frame = cv2.bitwise_and(next_frame, next_frame, mask=mask)
        cnt += 1

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

        good_p1 = p1[st == 1]
        good_p0 = p0[st == 1]

        n = np.linalg.norm(good_p1 - good_p0, axis=1)
        p = np.concatenate([good_p0, good_p1], axis=1)
        p = p[n > 2]

        if debug:
            for i in range(good_p0.shape[0]):
                next_frame = cv2.line(next_frame, (int(good_p0[i, 0]), int(good_p0[i, 1])),
                                      (int(good_p1[i, 0]), int(good_p1[i, 1])), (255, 0, 0), 1)
                next_frame = cv2.circle(next_frame, (int(good_p0[i, 0]), int(good_p0[i, 1])), 3, (0, 0, 255))
            cv2.imshow("Found", next_frame)
            cv2.waitKey(1)

        acc.accumulate_xy_lines(p)
        prev_gray = next_gray

        if cnt % 100 == 0:
            vp = acc.get_vp()
            print("VP so far: {}".format(vp))

    vp = acc.get_vp()
    return vp


def get_vp2(vp1, cap, mask, skip=10, debug=False):
    pp = [mask.shape[1] / 2 + 0.5, mask.shape[0] / 2 + 0.5]

    mask_reduced = cv2.erode(mask, element_big)

    back_sub = cv2.createBackgroundSubtractorMOG2()

    acc = Accumulator(size=256, debug=debug, height=mask.shape[0], width=mask.shape[1])

    cnt = -10
    ret = True

    while ret and cnt < 1000:
        for _ in range(skip):
            ret, frame = cap.read()
        if frame is None:
            continue
        cnt += 1
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        fg_mask = back_sub.apply(frame)
        # keep only best pts
        fg_mask = np.where(fg_mask > 127, 1.0, 0)
        fg_mask = cv2.erode(fg_mask, element_small)
        fg_mask = cv2.dilate(fg_mask, element_big)

        if cnt > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            # mag = mag / np.max(mag)

            # apply mask to image
            # gray = cv2.bitwise_and(mag, mag, mask=255*fg_mask.astype(np.uint8))

            mag_dilated = cv2.dilate(mag, element_loc_max)
            local_maxima = cv2.compare(mag, mag_dilated, cmpop=cv2.CMP_GE)
            # remove plateaus
            l_m_eroded = cv2.erode(local_maxima, element_loc_max)
            non_plateaus = cv2.compare(local_maxima, l_m_eroded, cmpop=cv2.CMP_GT)
            local_maxima_cleaned = cv2.bitwise_and(non_plateaus, local_maxima)

            seeds = np.logical_and(np.logical_and(np.logical_and(local_maxima_cleaned > 0, fg_mask > 0), mag > 450),
                                   mask_reduced > 0)
            seeds_idx = np.argwhere(seeds)

            a_list = []
            b_list = []
            c_list = []
            p = []

            for seed in seeds_idx:
                i, j = seed
                if 5 > i or i > frame.shape[0] - 5 or 5 > j or j > frame.shape[1] - 5:
                    continue
                window_x = sobel_x[i - 4:i + 5, j - 4:j + 5]
                window_y = sobel_y[i - 4:i + 5, j - 4:j + 5]
                matrix = np.column_stack([np.reshape(window_x, [81, 1]), np.reshape(window_y, [81, 1])])
                u, s, v = np.linalg.svd(matrix.T @ matrix)
                q = s[0] / s[1]
                d = u[:, 0]

                if q < 300:
                    continue

                n_vp = vp1 - np.flip(seed)
                dot_product = np.dot(n_vp / np.linalg.norm(n_vp), d / np.linalg.norm(d))
                angle = np.arccos(dot_product)

                if 0.325 * np.pi < angle < 0.625 * np.pi:
                    if debug:
                        frame = cv2.line(frame, (int(seed[1] - 10 * d[1]), int(seed[0] + 10 * d[0])),
                                         (int(seed[1] + 10 * d[1]), int(seed[0] - 10 * d[0])), (0, 0, 255), 1)
                        frame = cv2.line(frame, (int(vp1[0]), int(vp1[1])),
                                         (int(seed[1]), int(seed[0])), (0, 255, 255), 1)
                    continue

                a_list.append(d[0])
                b_list.append(d[1])
                c_list.append(- d[0] * seed[1] / 1920 - d[1] * seed[0] / 1080)

                # p.append([seed[1] - d[1], seed[0] + d[0], seed[1] + d[1], seed[0] - d[0]])

                if debug:
                    d = d / np.linalg.norm(d)
                    # print("Edgelet s:{}, q:{}, d:{}".format(seed, q, d))
                    frame = cv2.line(frame, (int(seed[1] - 10 * d[1]), int(seed[0] + 10 * d[0])),
                                     (int(seed[1] + 10 * d[1]), int(seed[0] - 10 * d[0])), (0, 255, 0), 1)

            acc.accumulate_abc_lines(np.array(a_list), np.array(b_list), np.array(c_list))
            # acc.accumulate_xy_lines(np.array(p))

            if debug:
                cv2.imshow("Found", frame)
                cv2.waitKey(1)

            if cnt % 100 == 0:
                vp = acc.get_conditional_vp(vp1, pp)
                print("VP so far: {}".format(vp))

    vp = acc.get_conditional_vp(vp1, pp)
    return vp


def calib_video(video_path, calib_path=None, debug=False, out_path=None):
    print('Calibrating for video: {}'.format(video_path))
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

    if calib_path is not None:
        with open(calib_path, 'r+') as file:
            structure = json.load(file)
            camera_calibration = structure['camera_calibration']
            vp1_test, vp2_test = camera_calibration["vp1"], camera_calibration["vp2"]
            print("Test VP: {}, {}".format(vp1_test, vp2_test))

    vp1 = get_vp1(cap, mask, debug=debug)
    print("Detected vp1: {}".format(vp1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vp2 = get_vp2(vp1, cap, mask, debug=debug, skip=3)
    print("Detected vp2: {}".format(vp2))

    if out_path is not None:
        pp = [mask.shape[1] / 2 + 0.5, mask.shape[0] / 2 + 0.5]
        camera_calibration = {'vp1': vp1, 'vp2': vp2, 'pp': pp}
        json_structure = {'cars': [], 'camera_calibration': camera_calibration}
        with open(out_path, 'w') as file:
            json.dump(json_structure, file)


if __name__ == "__main__":
    # vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
    # results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'
    #
    # vid_list = []
    # calib_list = []
    # for i in range(4, 7):
    #     dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
    #     # dir_list = ['session{}_left'.format(i), 'session{}_right'.format(i)]
    #     vid_list.extend([os.path.join(vid_path, d, 'video.avi') for d in dir_list])
    #     calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])
    #
    # for v, c in zip(vid_list, calib_list):
    #     test_video(v, c)

    # vid_dir = 'D:/Skola/PhD/data/DETRAC/Insight-MVT_Annotation_Test/'
    # vids = ['MVI_39031', 'MVI_39051', 'MVI_39211', 'MVI_39271', 'MVI_39371', 'MVI_39501', 'MVI_39511', 'MVI_40742',
    #         'MVI_40743', 'MVI_40863', 'MVI_40864']
    # # vids = ['MVI_40742', 'MVI_40743']
    #
    # vid_list = [os.path.join(vid_dir, v) for v in vids]
    # calib_list = [os.path.join(vid_dir, v, 'calib.json') for v in vids]
    #
    # for vid, calib in zip(vid_list, calib_list):
    #     calib_video(vid, debug=False, out_path=calib)

    calib_video('D:/Research/data/BASpeed/Zochova/video.m4v', out_path='D:/Research/data/BASpeed/Zochova/zochova.json', debug=True)
