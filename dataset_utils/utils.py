import cv2
import numpy as np
import os


def deprocess_image(x):
    x = np.array(x)
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    return x.astype(np.uint8)


def is_image(name):
    return '.jpg' in name or '.png' in name


class FolderVideoReader:
    def __init__(self, path):
        self.dir = path
        self.img_list = [img for img in os.listdir(path) if 'mask' not in img and is_image(img)]
        self.img_list = sorted(self.img_list)
        self.i = 0
        self.opened = True

    def read(self):
        if self.i + 1 >= len(self.img_list):
            self.opened = False
            return False, None

        img = cv2.imread(os.path.join(self.dir, self.img_list[self.i]))
        self.i += 1
        if img is None:
            self.opened = False
            return False, None
        return True, img

    def isOpened(self):
        return self.opened

    def release(self):
        ...

    def set(self, code, val):
        if code == cv2.CAP_PROP_POS_FRAMES:
            self.i = val
        return
