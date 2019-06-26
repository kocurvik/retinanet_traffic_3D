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
import cv2
import pickle

import numpy as np
import random
import threading
import warnings

from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path

import keras

from ..utils.anchors import (
    anchor_targets_bbox_centers,
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb, random_transform_generator


class Centers_Generator(object):

    def __init__(
        self,
        BCS_path,
        BoxCars_dataset,
        BoxCars_images,
        BCS_sessions = range(4),
        fake_centers = False,
        no_centers = False,
        batch_size=1,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=400,
        image_max_side=600,
        transform_list = None,
        transform_parameters = None,
        compute_anchor_targets=anchor_targets_bbox_centers,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """

        self.image_names = []
        self.image_data  = {}

        self.classes = {'car' : 0}

        # Take base_dir from annotations file if not explicitly specified.

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.fake_centers = fake_centers
        self.no_centers = no_centers

        self.image_data = {}
        self.transform_indices = []

        if BoxCars_dataset is not None:
            self.dataset_name = 'BoxCars'
            self.parse_BoxCars(BoxCars_dataset, BoxCars_images)

        for i in BCS_sessions:
            self.dataset_name = 'BCS'
            ds_path = os.path.join(BCS_path, 'dataset_{}.pkl'.format(i))
            im_path = os.path.join(BCS_path, 'images_{}'.format(i))
            self.parse_BCS(dataset_path=ds_path, images_path=im_path)

        print("Generator size: {}".format(self.size()))


        # self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side

        if transform_parameters is not None:
            self.transform_list = transform_list
        else:
            self.transform_list = [
                random_transform_generator(
                    min_translation=(-0.4, -0.4),
                    max_translation=(0.4, 0.4),
                    min_scaling=(0.9, 0.9),
                    max_scaling=(2.0, 2.0),
                    flip_x_chance=0.5
                ),
                random_transform_generator(
                    min_translation=(-0.5, -0.5),
                    max_translation=(0.5, 0.5),
                    min_scaling=(0.03, 0.03),
                    max_scaling=(1.0, 1.0),
                    flip_x_chance=0.5
                ),
            ]



        self.transform_parameters = transform_parameters or TransformParameters(fill_mode='constant')
        if self.no_centers:
            self.compute_anchor_targets = anchor_targets_bbox
        else:
            self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return self.image_names[image_index]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path   = self.image_names[image_index]
        annots = self.image_data[path]

        if self.no_centers:
            boxes = np.zeros((len(annots), 5))
        else:
            boxes  = np.zeros((len(annots), 6))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)
            if not self.no_centers:
                boxes[idx, 5] = float(annot['c'])

        return boxes

    def load_transform_indices(self, group):
        return [self.transform_indices[index] for index in group]

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            # if len(invalid_indices):
            #     cv2.imwrite("ID_.png".format(group[index]), image)
            #     warnings.warn("Following warning happens in:{}".format(self.dataset_name))
            #     warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
            #         group[index],
            #         image.shape,
            #         [annotations[invalid_index, :] for invalid_index in invalid_indices]
            #     ))
            #     annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform_index):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        transform_generator = self.transform_list[transform_index]
        if transform_generator:

            transform = adjust_transform_for_image(next(transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations, transform_index):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations, transform_index)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group, transform_indices):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations, transform_index) in enumerate(zip(image_group, annotations_group, transform_indices)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations, transform_index)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def generate_anchors(self, image_shape):
        return anchors_for_shape(image_shape, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        if self.no_centers:
            labels_batch, regression_batch, _ = self.compute_anchor_targets(
                anchors,
                image_group,
                annotations_group,
                self.num_classes()
            )
            return [regression_batch, labels_batch]
        else:
            labels_batch, regression_batch, centers_batch, _ = self.compute_anchor_targets(
                anchors,
                image_group,
                annotations_group,
                self.num_classes()
            )

            return [regression_batch, labels_batch, centers_batch]

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        transform_indices = self.load_transform_indices(group)
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group, transform_indices)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)

    def parse_BCS(self, dataset_path, images_path):
        with open(dataset_path, "rb") as f:
            ds = pickle.load(f, encoding='latin-1', fix_imports=True)

        for entry in ds:
            filename = os.path.join(images_path, entry['filename'])
            if filename not in self.image_data:
                self.image_data[filename] = []
                self.image_names.append(filename)
                self.transform_indices.append(0)
            if self.no_centers:
                for label in entry['labels']:
                    dict = {'x1': label['x_min'], 'x2': label['x_max'],
                            'y1': label['y_min'], 'y2': label['y_max'],
                            'class': 'car'}
                    self.image_data[filename].append(dict)
            elif self.fake_centers:
                for label in entry['labels']:
                    dict = {'x1': label['x_min'], 'x2': label['x_max'],
                            'y1': label['y_min'], 'y2': label['y_max'],
                            'c': 0.0, 'class': 'car'}
                    self.image_data[filename].append(dict)
            else:
                for label in entry['labels']:
                    dict = {'x1': label['x_min'], 'x2': label['x_max'],
                            'y1': label['y_min'], 'y2': label['y_max'],
                            'c' : label['centery'], 'class': 'car'}
                    self.image_data[filename].append(dict)

    def parse_BoxCars(self, dataset_path, images_path):
        with open(dataset_path, "rb") as f:
            ds = pickle.load(f, encoding='latin-1', fix_imports=True)
        for sample in ds['samples']:
            # to_camera = sample['to_camera']
            for i_id, instance in enumerate(sample['instances']):
                filename = os.path.join(images_path, instance['filename'])
                if filename not in self.image_data:
                    self.image_data[filename] = []
                    self.image_names.append(filename)
                    self.transform_indices.append(1)

                if self.no_centers:
                    dict = {'x1': instance['bb_out']['x_min'], 'x2': instance['bb_out']['x_max'],
                            'y1': instance['bb_out']['y_min'], 'y2': instance['bb_out']['y_max'],
                            'class': 'car'}
                else:
                    if self.fake_centers:
                        centery = 0.0
                    else:
                        centery = (instance['bb_in']['y_min'] - instance['bb_out']['y_min']) / \
                                  (instance['bb_out']['y_max'] - instance['bb_out']['y_min'])

                    dict = {'x1': instance['bb_out']['x_min'], 'x2': instance['bb_out']['x_max'],
                            'y1': instance['bb_out']['y_min'], 'y2': instance['bb_out']['y_max'],
                            'c': centery, 'class': 'car'}
                self.image_data[filename].append(dict)
