import json
import os
import cv2
import numpy as np

from dataset_utils.geometry import distance, computeCameraCalibration, getWorldCoordinagesOnRoadPlane
from scipy.spatial import Delaunay

import xml.etree.ElementTree as ET

from scipy.optimize import curve_fit

MAX_FRAMES = 200


def old_lane_functor():
    Y_LANE = 100
    Y_TOP = 50
    Y_BOTTOM = 600

    X_LANE12 = 600
    X_LANE23 = 900
    X_LANE34 = 1380

    def fit_line(y, k, l):
        return k * y + l

    def lane_fn(car):
        x = car['posX']
        y = car['posY']
        popt, _ = curve_fit(fit_line, y, x)
        x_lane = popt[0] * Y_LANE + popt[1]
        if x_lane < X_LANE12:
            return 1
        elif x_lane < X_LANE23:
            return 2
        elif x_lane < X_LANE34:
            return 3
        else:
            return 0

    def valid_meas_fn(car):
        pts = []
        frames = []
        for x, y, f in zip(car['posX'], car['posY'], car['frames']):
            if Y_TOP < y < Y_BOTTOM:
                pts.append(np.array([x, y]))
                frames.append(f)
        return np.array(pts), np.array(frames)

    return lane_fn, valid_meas_fn


SCALES = [[], [], []]


def projector_functor(matrices_path):
    matrices = np.loadtxt(matrices_path)

    m1 = matrices[:3, :]
    m2 = matrices[3:6, :]
    m3 = matrices[6:, :]

    def project_fn(pts, lane):
        x_h = np.array(pts).T
        if lane == 1:
            y = m1 @ x_h
        elif lane == 2:
            y = m2 @ x_h
        else:
            y = m3 @ x_h

        y = y.T
        y = y[:, :2] / y[:, 2, np.newaxis]
        return y

    return project_fn


def old_projector_functor(structure):
    vp1 = structure['camera_calibration']['vp1']
    vp2 = structure['camera_calibration']['vp2']
    pp = structure['camera_calibration']['pp']
    # scale = structure['camera_calibration']['scale']
    # vp1 = [660.0, -807.6190476190476]
    # vp2 = [-11988.0, 1344.0]

    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(vp1, vp2, pp)

    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    def project_fn(pts, lane):
        return np.array([getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp) for p in pts])

    return project_fn


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def lane_functor(path):
    root = ET.parse(path).getroot()[0]

    lane_hulls = [None, None, None]
    meas_hulls = [None, None, None]

    for lane in root:
        id = int(lane.attrib['id'])
        list = [[pts.attrib['x'], pts.attrib['y']] for pts in lane]
        if lane.tag == 'lane':
            lane_hulls[id - 1] = Delaunay(np.array(list))
        else:
            meas_hulls[id - 1] = Delaunay(np.array(list))

    def lane_fn(car):
        x = np.array(car['posX'])
        y = np.array(car['posY'])
        xy = np.column_stack([x, y])
        assigned_lanes = np.zeros_like(x)
        for i in range(3):
            cond = lane_hulls[i].find_simplex(xy) == 1
            assigned_lanes = np.where(cond, i + 1, assigned_lanes)

        counts = np.bincount(assigned_lanes.astype(np.int))
        if (counts[1:] > 5).any:
            counts[0] = 0
        return np.argmax(counts)

    def valid_meas_fn(car):
        x = np.array(car['posX'])
        y = np.array(car['posY'])
        xy = np.column_stack([x, y])
        frames = np.array(car['frames'])

        cond = lane_hulls[car['lane'] - 1].find_simplex(xy)
        return xy[cond > 0, :], frames[cond > 0]

    return lane_fn, valid_meas_fn


def compute_speed(car, valid_meas_fn, project_fn, scales):
    if car['lane'] == 0:
        return None
    xy, frames = valid_meas_fn(car)
    pts = np.column_stack([xy, np.ones_like(frames)])
    # pts = pts[-min(len(pts), MAX_FRAMES):]
    if len(pts) < 6:
        return None

    # coords = np.array([getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp) for p in pts])
    # coords = coords[:, :2] / coords[:, 2, np.newaxis]
    coords = project_fn(pts, car['lane'])

    # dists = np.linalg.norm(coords[:-5] - coords[5:], axis=1)
    # f_dists = frames[5:] - frames[:-5]
    dists = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
    f_dists = frames[1:] - frames[:-1]

    # dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    # f_dists = np.diff(np.array(frames))
    # median_dist = np.median(dists / f_dists)
    median_dist = np.mean(dists / f_dists)
    median_speed = median_dist * 3.6 / 30.15 * scales[car['lane'] - 1]
    return median_speed


def clean_gt(gt_cars):
    valid_gt_cars = []
    tot = 0
    for gt_car in gt_cars:
        radar_element = gt_car.find('radar')
        if radar_element is None or gt_car.attrib['moto'] == 'True' or gt_car.attrib['plate'] == 'False' or \
                gt_car.attrib['sema'] == 'True':
            continue
        radar = radar_element.attrib
        start_frame = int(radar['frame_start'])
        end_frame = int(radar['frame_end'])
        gt_lane = int(gt_car.attrib['lane'])
        gt_speed = float(radar['speed'])

        valid_gt_car = {'start_frame': start_frame, 'end_frame': end_frame, 'lane': gt_lane, 'speed': gt_speed,
                        'matched': False}
        valid_gt_cars.append(valid_gt_car)
    return valid_gt_cars


def track_iou(gt_car, car):
    frames = set(car['frames'])
    gt_frames = set(range(gt_car['start_frame'], gt_car['end_frame'] + 1))
    return len(frames.intersection(gt_frames)) / len(frames.union(gt_frames))


def match_cars(gt_cars, cars, verbose=False):
    errors = []
    valid_gt_cars = clean_gt(gt_cars)
    prev_assigned = 1

    # first match all easy matches to their own
    while sum([gt_car['matched'] for gt_car in valid_gt_cars]) != prev_assigned:
        prev_assigned = sum([gt_car['matched'] for gt_car in valid_gt_cars])
        for gt_car in valid_gt_cars:
            if gt_car['matched']:
                continue
            plausible_car_idxs = [idx for idx, car in enumerate(cars) if car['lane'] == gt_car['lane'] and
                                  (car['frames'][-min(len(car['frames']), MAX_FRAMES)] < gt_car['end_frame'] and
                                   gt_car['start_frame'] < car['frames'][-1])]
            if len(plausible_car_idxs) == 1:
                car = cars[plausible_car_idxs[0]]
                # errors.append(gt_car['speed'] - car['speed'])
                gt_car['matched'] = True
                gt_car['match'] = car
                del cars[plausible_car_idxs[0]]

    for gt_car in valid_gt_cars:
        if gt_car['matched']:
            continue

        plausible_car_idxs = [idx for idx, car in enumerate(cars) if car['lane'] == gt_car['lane'] and
                              (car['frames'][-min(len(car['frames']), MAX_FRAMES)] < gt_car['end_frame'] and
                               gt_car['start_frame'] < car['frames'][-1])]

        if len(plausible_car_idxs) == 0:
            continue
        elif len(plausible_car_idxs) == 1:
            car_idx = plausible_car_idxs[0]
        else:
            ious = [track_iou(gt_car, cars[idx]) for idx in plausible_car_idxs]
            max_idx = np.argmax(ious)
            car_idx = plausible_car_idxs[max_idx]
        car = cars[car_idx]
        # errors.append(gt_car['speed'] - car['speed'])
        gt_car['matched'] = True
        gt_car['match'] = car
        del cars[car_idx]

    return valid_gt_cars


def test_video(json_path, gt_path, scales, verbose=False):
    with open(json_path, 'r+') as file:
        structure = json.load(file)

    cars = structure['cars']

    gt_measurements_path = os.path.join(gt_path, 'vehicles.xml')
    gt_root = ET.parse(gt_measurements_path).getroot()
    gt_cars_xml = gt_root[0]
    gt_cars = []
    for car in gt_cars_xml:
        gt_cars.append(car)

    # lane_fn, valid_meas_fn = lane_functor(os.path.join(gt_path, 'lanes.xml'))
    lane_fn, valid_meas_fn = old_lane_functor()

    # project_fn = projector_functor(os.path.join(gt_path, 'matrix.txt'))

    project_fn = old_projector_functor(structure)

    valid_cars = []
    for car in cars:
        car['lane'] = lane_fn(car)
        speed = compute_speed(car, valid_meas_fn, project_fn, scales)
        if speed is not None:
            car['speed'] = speed
            valid_cars.append(car)

    matched_gt_cars = match_cars(gt_cars, valid_cars, verbose=verbose)

    if verbose:
        errors = [gt_car['speed'] - gt_car['match']['speed'] for gt_car in matched_gt_cars if gt_car['matched']]
        correct = ((errors < 2) & (errors > -3)).sum()

        print("Recall: {}".format(len(errors) / len(matched_gt_cars)))
        print("Correct from matched: {}".format(correct / len(errors)))
        print("Correct from all: {}".format(correct / len(matched_gt_cars)))

    return matched_gt_cars


if __name__ == "__main__":
    results_path = 'D:/Skola/PhD/data/LuvizonDataset/results/'
    # vid_dir = '/home/k/kocur15/data/luvizon/videos/'

    if os.name == 'nt':
        vid_path = 'D:/Skola/PhD/data/LuvizonDataset/dataset/'
        results_path = 'D:/Skola/PhD/data/LuvizonDataset/results/'
    else:
        vid_path = '/home/k/kocur15/data/luvizon/dataset/'
        results_path = '/home/k/kocur15/data/luvizon/results/'

    vid_dict = {1: [1, 2], 2: [1, 2, 3, 4, 5, 6], 3: [1], 4: [1], 5: [1]}
    # vid_dict = {1: [1], 2: [1], 3: [1], 4: [1], 5: [1]}
    name = 'system_Transform3D_BCL_0.5_960_540_VP2VP3.json'
    # name = 'system_Transform3D_960_540_VP2VP3.json'
    # name = 'system_Transform3D_BCL_0.5_640_360_VP2VP3.json'

    # name = 'system_Transform3D_960_540_VP2VP3.json'
    # vid_dict = {1: [1, 2, 3, 4], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3: [1, 2], 4: [1, 2], 5: [1]}

    calib_list = []
    gt_list = []

    for i in vid_dict.keys():
        gt_list.extend([os.path.join(vid_path, 'subset0{}'.format(i), 'video{:02d}'.format(j)) for j in vid_dict[i]])
        calib_list.extend(
            [os.path.join(results_path, 'subset0{}'.format(i), 'video{:02d}'.format(j), name) for j in vid_dict[i]])

    scales = np.array([22.64256336, 22.02260989, 20.87040733])

    results = []
    for calib, gt in zip(calib_list, gt_list):
        results.extend(test_video(calib, gt, scales))

    errors = np.array([gt_car['speed'] - gt_car['match']['speed'] for gt_car in results if gt_car['matched']])
    correct = ((errors < 2) & (errors > -3)).sum()

    print("Recall: {}".format(len(errors) / len(results)))
    print("Correct from matched: {}".format(correct / len(errors)))
    print("Correct from all: {}".format(correct / len(results)))
