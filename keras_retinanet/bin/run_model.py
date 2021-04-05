import sys
import os

import cv2
import numpy as np

from keras_retinanet.preprocessing.centers_generator import Centers_Generator

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

from .. import models

if __name__ == '__main__':
    # model = models.load_model('./snapshots/mobilenet224_1_BC+BCS_10.h5', backbone_name='mobilenet224_1')

    model = models.load_model('./snapshots/converted.h5', backbone_name='mobilenet224_1')

    print(model.summary())

    BCS_path = 'D:/Skola/PhD/data/BCS_boxed/'

    # BCS_path = '/home/kocur/data/BCS_boxed/'

    Box_images = 'C:/datasets/BoxCars116k/images_warped'
    Box_dataset = 'C:/datasets/BoxCars116k/dataset_warped.pkl'

    # Box_images = '/home/kocur/data/BoxCars116k/images_warped/'
    # Box_dataset = '/home/kocur/data/BoxCars116k/dataset_warped.pkl'

    train_generator = Centers_Generator(
        BCS_path,
        None,
        None,
        BCS_sessions=[0],
        batch_size=1
    )

    entry = next(train_generator)

    image = entry[0][0]

    cv2.imshow('Lol', image)

    cv2.waitKey(100)

    y_pred = model.predict(np.expand_dims(image, axis=0), 1)

    print(y_pred[0].shape)
    print(y_pred[1].shape)
    print(y_pred[2].shape)

    boxes = np.concatenate([y_pred[1][0, :, None], y_pred[0][0, :, :], y_pred[2][0, :, None]], 1)

    print(boxes)
