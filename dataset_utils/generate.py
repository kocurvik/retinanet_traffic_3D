import pickle
import numpy as np
import os
import sys
import cv2
from warper import warp_generator


# Script to generate the dataset from BoxCars116k for Transform2D and Transform3D approaches from the paper

def generate_warped_dataset(pair, dataset_path, images_path, t_dataset_path, t_images_path, im_w, im_h):
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f, encoding='latin-1', fix_imports=True)

    t_dataset_path = t_dataset_path.format(pair)
    t_images_path = t_images_path.format(pair)

    unacceptable = ['uvoz', 'prahaVinohradska', 'stefanikova', 'videnska']

    tds = {'samples': []}

    for s_id, sample in enumerate(ds['samples']):

        camera = sample['camera']
        if camera in unacceptable:
            continue
        t_sample = {'instances': [], 'id': s_id, 'to_camera': sample['to_camera'], 'annotation': sample['annotation']}
        t_sample.update({'id': s_id})
        for instance in sample['instances']:
            if not sample['to_camera']:
                continue
            path = os.path.join(images_path, instance['path'])
            image = cv2.imread(path)
            targetpath = os.path.join(t_images_path, instance['path'])
            bb3d = instance['3DBB'] - instance['3DBB_offset']

            vp1 = ds['cameras'][camera]['vp1'] - instance['3DBB_offset']
            vp2 = ds['cameras'][camera]['vp2'] - instance['3DBB_offset']
            vp3 = ds['cameras'][camera]['vp3'] - instance['3DBB_offset']
            # if not sample['to_camera']:
            #     vp2, vp3 = vp3, vp2

            if pair == '12':
                t_image, _, bb_in, bb_out = warp_generator(image, bb3d, vp1, vp2, im_h, im_w)
            elif pair == '13':
                t_image, _, bb_in, bb_out = warp_generator(image, bb3d, vp1, vp3, im_h, im_w)
            else:
                t_image, _, bb_in, bb_out = warp_generator(image, bb3d, vp3, vp2, im_h, im_w)

            if not os.path.exists(os.path.dirname(targetpath)):
                os.makedirs(os.path.dirname(targetpath))

            print(targetpath)
            print(cv2.imwrite(targetpath, t_image))

            # print(bb_in)
            #
            # print(bb_out)
            #
            # print(vp0_t)

            t_instance = {'filename': instance['path'], 'bb_in': bb_in, 'bb_out': bb_out,
                          'instance_id': instance['instance_id']}
            t_sample['instances'].append(t_instance)
        tds['samples'].append(t_sample)

    with open(t_dataset_path, "wb") as f:
        pickle.dump(tds, f)


if __name__ == "__main__":
    # generate_warped_dataset('12','C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/dataset_warped{}.pkl', 'C:/datasets/BoxCars116k/images_warped{}/', 300,
    #                         300)

    generate_warped_dataset('12', '/home/k/kocur15/data/BoxCars116k/dataset.pkl',
                            '/home/k/kocur15/data/BoxCars116k/images/',
                            '/home/k/kocur15/data/BoxCars116k/dataset_warped_rot{}.pkl',
                            '/home/k/kocur15/data/BoxCars116k/images_warped_rot{}/',
                            180, 320)

    # generate_warped_dataset('/home/kocur/data/BoxCars116k/dataset.pkl', '/home/kocur/data/BoxCars116k/images/',
    #                         '/home/kocur/data/BoxCars116k/dataset_warped.pkl', '/home/kocur/data/BoxCars116k/images_warped/', 320,
    #                         180)