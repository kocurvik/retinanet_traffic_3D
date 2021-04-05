import math

import numpy as np


# Couple of helper geometry functions
# Last two of them are taken from https://github.com/JakubSochor/BrnoCompSpeed

def distance(p1, p2, p3):
    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)


def is_right(l1, l2, p):
    return ((p[0] - l1[0]) * (l2[1] - l1[1]) - (p[1] - l1[1]) * (l2[0] - l1[0])) < 0


def isLeft(A, B, P):
    ret = (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])
    return ret < 0


def tangent_point_poly(p, V, im_h):
    left_idx = 0
    right_idx = 0
    p = [np.float64(x) for x in p]
    n = len(V)
    for i in range(1, n):
        if isLeft(p, V[left_idx], V[i]):
            left_idx = i
        if not isLeft(p, V[right_idx], V[i]):
            right_idx = i
    if p[1] > im_h:
        return V[left_idx], V[right_idx]
    return V[right_idx], V[left_idx]


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return np.float32(x), np.float32(y)
    else:
        return False


def getFocal(vp1, vp2, pp):
    return math.sqrt(-np.dot(vp1[0:2] - pp[0:2], vp2[0:2] - pp[0:2]))


def getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp):
    p = p / p[2]
    pp = pp / pp[2]
    ppW = np.concatenate((pp[0:2], [0]))
    pW = np.concatenate((p[0:2], [focal]))
    dirVec = pW - ppW
    t = -np.dot(roadPlane, np.concatenate((ppW, [1]))) / np.dot(roadPlane[0:3], dirVec)
    return ppW + t * dirVec


def computeCameraCalibration(_vp1, _vp2, _pp):
    vp1 = np.concatenate((_vp1, [1]))
    vp2 = np.concatenate((_vp2, [1]))
    pp = np.concatenate((_pp, [1]))
    focal = getFocal(vp1, vp2, pp)
    vp1W = np.concatenate((_vp1, [focal]))
    vp2W = np.concatenate((_vp2, [focal]))
    ppW = np.concatenate((_pp, [0]))
    vp3W = np.cross(vp1W - ppW, vp2W - ppW)
    vp3 = np.concatenate((vp3W[0:2] / vp3W[2] * focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal])) - ppW
    roadPlane = np.concatenate((vp3Direction / np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal
