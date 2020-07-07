import json

import numpy as np
import os
import sys
import cv2

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ))
    print(sys.path)


from dataset_utils.tracker import Tracker
from dataset_utils.warper import get_transform_matrix_with_criterion
from dataset_utils.geometry import computeCameraCalibration


def track_detections(json_path, video_path, pair,  im_w, im_h, name, threshold, fake = False, write_name = None, keep = 5):
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

    if pair == '12':
        M, IM = get_transform_matrix_with_criterion(vp1, vp2, mask, im_w, im_h)
    elif pair == '23':
        M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h)

    tracker = Tracker(json_path, M, IM, vp1, vp2, vp3, im_w, im_h, name, threshold=threshold, pair = pair, fake=fake, write_name=write_name, keep = keep)
    tracker.read()

if __name__ == "__main__":
    # vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
    # results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'

    vid_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/dataset/'
    results_path = '/home/k/kocur15/data/2016-ITS-BrnoCompSpeed/results/'

    vid_list = []
    calib_list = []
    for i in range(4, 7):
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        vid_list.extend([os.path.join(vid_path, d) for d in dir_list])
        calib_list.extend([os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])

    cases = []
    # cases.append({'width': 480, 'height': 270, 'pair': '23', 'name': '{}_{}_{}_1_at30', 'fake': False})
    # cases.append({'width': 640, 'height': 360, 'pair': '23', 'name': '{}_{}_{}_3_at30', 'fake': False})
    # cases.append({'width': 960, 'height': 540, 'pair' :'23', 'name': '{}_{}_{}_1_at30', 'fake': False})
    # cases.append({'width': 270, 'height': 480, 'pair': '12', 'name': '{}_{}_{}_1', 'fake': False})
    # cases.append({'width': 360, 'height': 640, 'pair': '12', 'name': '{}_{}_{}_1', 'fake': False})
    # cases.append({'width': 540, 'height': 960, 'pair' :'12', 'name': '{}_{}_{}_1', 'fake': False})

    # cases.append({'width': 640, 'height': 360, 'pair': '23', 'name': '{}_{}_no_centers_{}_0_at30', 'fake': True})
    # cases.append({'width': 360, 'height': 640, 'pair': '12', 'name': '{}_{}_no_centers_{}_0_at30', 'fake': True})
    #
    # cases.append({'width': 640, 'height': 360, 'pair': '23', 'name': 'mask_ablation', 'fake': False})


    VP_string = {'12': 'VP1VP2', '23': 'VP2VP3'}


    write_names = []
    for case in cases:
        if case['fake']:
            write_name = 'Transform2D_{}_{}_{}'.format(case['width'], case['height'], VP_string[case['pair']])
        else:
            if case['name'] == 'mask_ablation':
                write_name = 'MaskRCNN_1024_576'
            else:
                write_name = 'Transform3D_{}_{}_{}'.format(case['width'], case['height'], VP_string[case['pair']])

        name = case['name'].format(case['width'], case['height'], case['pair'])
        write_names.append('system_{}'.format(write_name))
        print("Performing tracking for: {} saving to: {}".format(name, write_name))
        for calib, vid in zip(calib_list, vid_list):
            print("Video: {}".format(vid))
            track_detections(calib, vid, case['pair'], case['width'], case['height'], name=name,
                             threshold=0.5, fake=case['fake'], keep=10, write_name=write_name)
    print(write_names)
    print(tuple(write_names))


