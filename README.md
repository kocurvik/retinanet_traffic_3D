# Implementation for paper: "Detection of 3D bounding boxes of vehicles using perspective transformation for accurate speed measurement" in Machine Vision and Applications

Link to the [paper](https://rdcu.be/b6Ofz).

This work is an extension of our previous work titled: "Perspective transformation for accurate detection of 3D bounding boxes of vehicles in traffic surveillance" presented at CVWW 2019 [link](https://openlib.tugraz.at/download.php?id=5c5941d91c84c&location=browse).

# :warning: UPDATE: New repo based on the same approach with YOLOv6

We have reimpelemted the method to use a more modern object detector YOLOv6 and performed extensive evaluation in our follow-up paper titled [Efficient vision-based vehicle speed estimation](https://link.springer.com/article/10.1007/s11554-025-01704-z). The code can be found in [this repo](https://github.com/gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D).

You can still use this repo for your purposes, but it requires tf 1.14 to run properly which might be difficult to install. The computational efficiency of the newer version is much better while having virtually the same speed measurement accuracy.

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
| Transform 3D  | 23 | 480 x 270 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQCtIMOnw1-4TI6hVvq5-K89AYvXXrZ0YxO9s4NW7fBqtVw?e=6dRZOK) |
| Transform 3D  | 23 | 640 x 360 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQDCxO5cCLKXSbNJMtr2rtHdAa-H_xDrC1r6JmjJIZ4KI4k?e=cOUZhp) |
| Transform 3D  | 23 | 960 x 540 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQAcyz2xTN4BT7wHGXSi-NDDAUMBP4p2nYSXx96H42wuUdM?e=wYIsX3) |
| Transform 3D  | 12 | 270 x 480 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQCchgh7PFP7Ro_TjIFaK0joAavmwRBeWDZBfLc-2X3ci8k?e=KRbkZd)|
| Transform 3D  | 12 | 360 x 640 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQApljtR_MCWTbjHMF7G7coOAbXqcx7GrK495RL_jhdC4-E?e=lH0230) |
| Transform 3D  | 12 | 540 x 960 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQCByPxqiih7Qba980N50xarATLf41QjvuRRN2g5GiDyQV0?e=VplKnY)|
| Transform 2D (ablation)  | 23 | 640 x 360 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQCjA3UAFEb3SZBsvozE7aPQAdVuRqNwLnDeoFJB5l4Xj0k?e=wPnSIo) |
| Orig 2D (ablation) | - | 640 x 360 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQD1-TmfKwlRQZDqLKeFBXOIAdagAvADZzPDBEC6u3iAaOo?e=U6GB4g) |
| Transform 3D (luvizon et al. dataset) | 23 | 640 x 360 | [download](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/IQB2vEgQ2qhLR5sH2SS5cjFjAZPIfbx34x5bzscfdbS1Mvk?e=tKru85) |

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
