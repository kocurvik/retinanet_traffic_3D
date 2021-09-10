# Implementation for paper: "Detection of 3D bounding boxes of vehicles using perspective transformation for accurate speed measurement" in Machine Vision and Applications

Link to the [paper](https://rdcu.be/b6Ofz).

This work is an extension of our previous work titled: "Perspective transformation for accurate detection of 3D bounding boxes of vehicles in traffic surveillance" presented at CVWW 2019 [link](https://openlib.tugraz.at/download.php?id=5c5941d91c84c&location=browse).

## Acknowledgment

This repository is a fork of the wonderful keras-retinanet repo: https://github.com/fizyr/keras-retinanet

Note that this repository was uploaded upon an individual request a year after the final experiments have been performed. It might not work as it is difficult to easily install TF 1.14. Also there may have been some code changes which aimed at producing some of the paper figures which might have broken the code.

## Installation

1) Install tensorflow 1.14, keras 2.2.3, keras-resnet, cython, six, opencv, numpy
2) If you want to train models you need to also install https://github.com/matterport/Mask_RCNN in a separate directory  
3) You can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.

## Testing

All of the code related to testing is located in the dataset_utils directory.

To run the models you will need to download the relevant model and run dataset_utils/test.py You can find further instructions in that file.

The models are run on the BrnoCompSpeed dataset videos and also require the calibration files from the relevant papers. The website (http://medusa.fit.vutbr.cz) of the research group hosting the dataset and the calibration files has been down since summer 2020. If you obtain the BrnoCompSpeed dataset you can use the calibration files which are provided in my [repository](https://github.com/kocurvik/BCS_results) mentioned in the paper. To evaluate the results please refer to the BrnoCompSpeed evaluation code [repository](https://github.com/JakubSochor/BrnoCompSpeed)

### Model zoo

| Type        | Pair | Resolution           | Model  |
| ------------- | ------ |-------------| -----|
| Transform 3D  | 23 | 480 x 270 | [download](https://drive.google.com/file/d/1O6mjzdrgPwg8J9XxmWsFnWxQHXmF_nR0/view?usp=sharing) |
| Transform 3D  | 23 | 640 x 360 | [download](https://drive.google.com/file/d/1SERwZojQL_Efaq5WeROEmKt5LpyQ0h5x/view?usp=sharing) |
| Transform 3D  | 23 | 960 x 540 | [download](https://drive.google.com/file/d/1LhgKrujz9kgOrv33rOSWJIhNQalK4ug_/view?usp=sharing) |
| Transform 3D  | 12 | 270 x 480 | [download](https://drive.google.com/file/d/1MGamEp4o5QoBbpK0lcTfQ0xNjC0D9jNh/view?usp=sharing)|
| Transform 3D  | 12 | 360 x 640 | [download](https://drive.google.com/file/d/1pvwbTDNMLJrS9MGE-p4QxcyoHdve0MqO/view?usp=sharing) |
| Transform 3D  | 12 | 540 x 960 | [download](https://drive.google.com/file/d/1qVpq1TG93Ae-xO2jF-Bigiyvfjdjy06C/view?usp=sharing) |
| Transform 2D (ablation)  | 23 | 640 x 360 | [download](https://drive.google.com/file/d/1ea5ia4u927jwvEEVDhHiZAKEdxdb7Bwr/view?usp=sharing) |
| Orig 2D (ablation) | - | 640 x 360 | [download](https://drive.google.com/file/d/1Koq4x16K4-4tOLpOjhqASntcTQWkCkvz/view?usp=sharing) |
| Transform 3D (luvizon et al. dataset) | 23 | 640 x 360 | [download](https://drive.google.com/file/d/1MlW1I1-INaBu4HAdH0qtQyfIflfCni3x/view?usp=sharing) |

## Training

To train the models you have to run the scripts which produce the training data. Change the parameters in function calls in the `main` body of the script accordingly and run `dataset_utils/fromBrnoCompSpeed.py` and `dataset_utils/generate.py`

Then you can run the the training script for example:

```
python keras_retinanet/bin/train.py --no-evaluation --backbone resnet50 --epochs 30 --steps 10000 --batch-size 16 --pair 23 --image-min-side 360 --image-max-side 640 --centers --gpu 0 --snapshot-path ./snapshots/640_360_23_0 --tensorboard-dir ./logs/640_360_23_0 BC+BCS
```

Afterwards you have to convert the model:

```
python keras_retinanet/bin/convert_model.py --backbone resnet50 snapshots/BC+BCS_640_360_23_0/resnet50_BC+BCS_30.h5 snapshots/640_360_23/resnet50_640_360_23.h5
```

## Citations

If you find this repository useful please consider citing the related publications:

```
﻿@article{kocurMVAP2020,
  author={Kocur, Viktor and Ft{\'a}{\v{c}}nik, Milan},
  title={Detection of 3D bounding boxes of vehicles using perspective transformation for accurate speed measurement},
  journal={Machine Vision and Applications},
  year={2020},
  month={Sep},
  day={04},
  volume={31},
  number={7},
  pages={62},
  issn={1432-1769},
  doi={10.1007/s00138-020-01117-x}
}
```



```
@inproceedings{kocurCVWW2019,
  title={Perspective transformation for accurate detection of 3D bounding boxes of vehicles in traffic surveillance.},
  author={Kocur, Viktor},
  booktitle={Proceedings of the 24th Computer Vision Winter Workshop},
  pages={33-41},
  year={2019},
  doi={10.3217/978-3-85125-652-9-04}
}
```
