#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import cv2

# Allow relative imports when being executed as script.
from keras_retinanet.preprocessing.centers_generator import Centers_Generator

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes
from ..utils.anchors import anchors_for_shape


def create_generator():
    """ Create the data generators.
    """

    transform_generator = random_transform_generator(flip_x_chance=0.0)

    BCS_path = 'D:/Skola/PhD/data/BCS_boxed/'

    Box_images = 'C:/datasets/BoxCars116k/images_warped'
    Box_dataset = 'C:/datasets/BoxCars116k/dataset_warped.pkl'

    common_args = {
        'batch_size': 1,
        'image_min_side': 540,
        'image_max_side': 960
    }

    generator = Centers_Generator(
        BCS_path,
        Box_dataset,
        Box_images,
        BCS_sessions=[0],
        **common_args
    )

    return generator


def run(generator):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image = generator.load_image(i)
        annotations = generator.load_annotations(i)
        transform_index = generator.transform_indices[i]

        # apply random transformations
        image, annotations = generator.random_transform_group_entry(image, annotations, transform_index)

        image, image_scale = generator.resize_image(image)
        annotations[:, :4] *= image_scale

        anchors = anchors_for_shape(image.shape)

        labels_batch, regression_batch, centers_batch, boxes_batch = generator.compute_anchor_targets(anchors, [image],
                                                                                                      [annotations],
                                                                                                      generator.num_classes())
        anchor_states = labels_batch[0, :, -1]

        # draw anchors on the image
        # if args.anchors:
        draw_boxes(image, boxes_batch[0, anchor_states == 1, :], (0, 255, 0), thickness=1)

        # draw annotations on the image
        # if args.annotations:
        # draw annotations in red
        draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

        # draw regressed anchors in green to override most red annotations
        # result is that annotations without anchors are red, with anchors are green
        draw_boxes(image, boxes_batch[0, anchor_states == 1, :], (0, 255, 0))

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True


def main(args=None):
    # parse arguments
    generator = create_generator()

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    while run(generator):
        pass
    # else:
    #     run(generator)


if __name__ == '__main__':
    main()
