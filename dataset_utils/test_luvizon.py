import json
import os
import cv2
import numpy as np

from dataset_utils.geometry import distance, computeCameraCalibration, getWorldCoordinagesOnRoadPlane

import xml.etree.ElementTree as ET

from scipy.optimize import curve_fit

Y_LANE = 100
Y_TOP = 50
Y_BOTTOM = 300

X_LANE12 = 600
X_LANE23 = 900
X_LANE34 = 1380

LANE1_MATRIX = np.array([[0.45748904,	0.45510265,	95.80971527],
                         [0.04026574,	1.66259563,	-5.53627539],
                         [-0.00011619,	0.00107257,	1.00000000]])

LANE2_MATRIX = np.array([[0.45887539, 0.70749354, 337.20050049],
                         [0.03581188, 1.72447097, -7.71867371],
                         [-0.00004267, 0.00108679, 1.00000000]])

LANE3_MATRIX = np.array([[0.37196219, 0.92335510, 589.98767090],
                         [-0.03626568, 1.70345390, 54.57832336],
                         [-0.00008483, 0.00105958, 1.00000000]])



MAX_FRAMES = 25

SCALES = []

def project(pts, lane):
    x_h = np.array(pts).T
    if lane == 1:
        y = LANE1_MATRIX @ x_h
    elif lane == 2:
        y = LANE2_MATRIX @ x_h
    else:
        y = LANE3_MATRIX @ x_h

    y = y.T
    y = y[:, :2]/y[:, 2, np.newaxis]
    return y

def fit_line(y, k, l):
    return k*y + l


def compute_speed(car, focal, roadPlane, pp, scale):
    pts = []
    frames = []
    for x, y, f in zip(car['posX'], car['posY'], car['frames']):
        if Y_TOP < y < Y_BOTTOM:
            pts.append(np.array([x, y, 1]))
            frames.append(f)

    # pts = pts[-min(len(pts), MAX_FRAMES):]
    if len(pts) < 6:
        return 50.0

    # coords = np.array([getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp) for p in pts])
    # coords = coords[:, :2] / coords[:, 2, np.newaxis]
    coords = project(pts, car['lane'])

    dists = np.linalg.norm(coords[:-5] - coords[5:], axis=1)
    frames = np.array(frames)
    f_dists = frames[5:] - frames[:-5]

    # dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    # f_dists = np.diff(np.array(frames))
    median_dist = np.median(dists/f_dists)
    median_speed = median_dist * scale * 3.6

    return median_speed


def match_cars(gt_cars, cars):
    errors = []
    for gt_car in gt_cars:

        radar_element = gt_car.find('radar')
        if radar_element is None or gt_car.attrib['moto'] == 'True':
            continue
        radar = radar_element.attrib
        start_frame = int(radar['frame_start'])
        end_frame = int(radar['frame_end'])
        gt_lane = int(gt_car.attrib['lane'])
        gt_speed = float(radar['speed'])

        # print(radar)

        plausible_cars = [car for car in cars if car['lane'] == gt_lane
                          and (car['frames'][-min(len(car['frames']), 100)] < end_frame and start_frame < car['frames'][-1])]

        # for car in plausible_cars:
        #     print("Detection speed: {} in lane {}".format(car['speed'], car['lane']))
        if len(plausible_cars) == 1:
            SCALES.append(gt_speed / plausible_cars[0]['speed'])
            errors.append(gt_speed - plausible_cars[0]['speed'])
        elif len(plausible_cars) > 1:
            ers = [gt_speed - car['speed'] for car in plausible_cars]
            idx = np.argmin(np.abs(ers))
            # SCALES.append(gt_speed/plausible_cars[idx]['speed'])
            errors.append(ers[idx])

    errors = np.array(errors)

    print("Recall: {}".format(len(errors)/ len(gt_cars)))
    print("Correct: {}".format(((errors < 3) & (errors > -3)).sum() / len(errors)))
    print(np.median(np.abs(errors)))
    print(np.mean(np.abs(errors)))


def test_video(results_path, name):
    json_path = os.path.join(results_path, 'system_{}.json'.format(name))
    with open(json_path, 'r+') as file:
        structure = json.load(file)

    vp1 = structure['camera_calibration']['vp1']
    vp2 = structure['camera_calibration']['vp2']
    pp = structure['camera_calibration']['pp']
    # scale = structure['camera_calibration']['scale']
    # vp1 = [660.0, -807.6190476190476]
    # vp2 = [-11988.0, 1344.0]
    scale = 1.0 * 0.557577870532032



    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(vp1, vp2, pp)

    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    cars = structure['cars']

    for car in cars:
        x = car['posX']
        y = car['posY']
        popt, _ = curve_fit(fit_line, y, x)
        x_lane = popt[0] * Y_LANE + popt[1]
        if x_lane < X_LANE12:
            lane = 1
        elif x_lane < X_LANE23:
            lane = 2
        elif x_lane < X_LANE34:
            lane = 3
        else:
            lane = 4
        car['lane'] = lane
        car['speed'] = compute_speed(car, focal, roadPlane, pp, scale)


    gt_root = ET.parse(os.path.join(results_path, 'gt.xml')).getroot()
    gt_cars_xml = gt_root[0]
    gt_cars = []
    for car in gt_cars_xml:
        gt_cars.append(car)

    matches = match_cars(gt_cars, cars)



if __name__ == "__main__":
    res_dir = 'D:/Skola/PhD/data/LuvizonDataset/results/'
    # vid_dir = '/home/k/kocur15/data/luvizon/videos/'

    name = 'Transform3D_960_540_VP2VP3'

    for i in range(1, 6):
        result_dir = os.path.join(res_dir, 'Set0{}'.format(i))
        test_video(result_dir, name)


    print(np.median(SCALES))