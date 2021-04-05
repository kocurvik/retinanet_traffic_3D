import numpy as np
import cv2
from dataset_utils.geometry import computeCameraCalibration


class Accumulator:
    def __init__(self, size=512, width=1920, height=1080, debug=False):
        self.size = size
        self.ds = np.zeros([2 * self.size + 1, 2 * self.size + 1], dtype=np.int32)
        self.debug = debug
        self.width = width
        self.height = height

    def accumulate_xy_lines(self, lines):
        # normalize lines
        lines[:, 0] /= self.height
        lines[:, 2] /= self.height
        lines[:, 1] /= self.width
        lines[:, 3] /= self.width

        p1 = lines[:, 2:]
        p2 = lines[:, :2]

        a = p1[:, 1] - p2[:, 1]
        b = p2[:, 0] - p1[:, 0]
        c = - a * p1[:, 0] - b * p1[:, 1]

        self.accumulate_abc_lines(a, b, c)

    def accumulate_abc_lines(self, a, b, c):
        alpha = np.where(a * b >= 0, 1.0, -1.0)
        beta = np.where(b * c >= 0, 1.0, -1.0)
        gamma = np.where(a * c >= 0, 1.0, -1.0)

        edgepts = np.empty([a.shape[0], 8], dtype=np.float32)
        edgepts[:, 0] = alpha * a / (c + gamma * a)
        edgepts[:, 1] = -alpha * c / (c + gamma * a)
        edgepts[:, 2] = b / (c + beta * b)
        edgepts[:, 3] = 0.0
        edgepts[:, 4] = 0.0
        edgepts[:, 5] = b / (a + alpha * b)
        edgepts[:, 6] = -alpha * a / (c + gamma * a)
        edgepts[:, 7] = alpha * c / (c + gamma * a)

        edgepts = np.round(edgepts * self.size) + self.size

        for segments in edgepts:
            self.add_line_segments(segments)
            if self.debug:
                cv2.imshow("space", np.log(self.ds) / np.log(np.max(self.ds)))
                cv2.waitKey(1)
        return

    def add_line_segments(self, p):
        canvas = np.zeros_like(self.ds, dtype=np.uint8)
        canvas = cv2.line(canvas, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1)
        canvas = cv2.line(canvas, (int(p[2]), int(p[3])), (int(p[4]), int(p[5])), 1)
        canvas = cv2.line(canvas, (int(p[4]), int(p[5])), (int(p[6]), int(p[7])), 1)
        self.ds += canvas

    def get_vp(self):
        q, p = np.unravel_index(np.argmax(self.ds), self.ds.shape)
        q = (q - self.size) / self.size
        p = (p - self.size) / self.size
        # print(p, q)
        vp = [q, np.sign(p) * p + np.sign(q) * q - 1, p]
        return self.height * vp[0] / vp[2], self.width * vp[1] / vp[2]

    def get_conditional_vp(self, vp1, pp):
        success = False
        ds = self.ds.copy()
        while not success:
            idx = np.argmax(ds)
            q_idx, p_idx = np.unravel_index(idx, ds.shape)
            q = (q_idx - self.size) / self.size
            p = (p_idx - self.size) / self.size
            # print(p, q)
            vp2 = [q, np.sign(p) * p + np.sign(q) * q - 1, p]
            vp2 = [self.height * vp2[0] / vp2[2], self.width * vp2[1] / vp2[2]]
            success = True

            try:
                computeCameraCalibration(vp1, vp2, pp)
            except ValueError:
                print('Failed vp2: {}'.format(vp2))
                success = False
                ds[q_idx, p_idx] = 0
                continue

        return vp2
