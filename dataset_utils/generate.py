import pickle
import numpy as np
import os
import sys
import cv2
from dataset_utils.warper import warp_generator

def generate_warped_dataset(dataset_path, images_path, t_dataset_path, t_images_path, im_w, im_h):
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f, encoding='latin-1', fix_imports=True)

    # badcameras = ['fit', 'videnska', 'kostel', 'fitDistorted', 'prahaVinohradska', 'kostelDistorted', 'stefanikova',
    #               'uvoz']

    unacceptable = ['uvoz', 'prahaVinohradska', 'stefanikova', 'videnska']


    tds = {'samples': []}

    for s_id, sample in enumerate(ds['samples']):

        # if sample['id'] != 26886:
        #     continue
        # sample = ds['samples'][24444]
        camera = sample['camera']
        if camera in unacceptable:
            continue
        t_sample = {'instances': [], 'id': s_id, 'to_camera': sample['to_camera'], 'annotation': sample['annotation']}
        t_sample.update({'id': s_id})
        for instance in sample['instances']:
            path = os.path.join(images_path, instance['path'])
            image = cv2.imread(path)
            targetpath = os.path.join(t_images_path, instance['path'])
            bb3d = instance['3DBB'] - instance['3DBB_offset']

            vp1 = ds['cameras'][camera]['vp1'] - instance['3DBB_offset']
            vp2 = ds['cameras'][camera]['vp2'] - instance['3DBB_offset']
            vp3 = ds['cameras'][camera]['vp3'] - instance['3DBB_offset']
            if not sample['to_camera']:
                vp2, vp3 = vp3, vp2

            # t_image, _, bb_in, bb_out = warp_generator(image, bb3d, vp1, vp2, im_h, im_w)
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
    # generate_warped_dataset('C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/dataset_warped23.pkl', 'C:/datasets/BoxCars116k/images_warped23/', 300,
    #                         300)

    generate_warped_dataset('/home/k/kocur15/data/BoxCars116k/dataset.pkl', '/home/k/kocur15/data/BoxCars116k/images/',
                            '/home/k/kocur15/data/BoxCars116k/dataset_warped23.pkl', '/home/k/kocur15/data/BoxCars116k/images_warped23/',
                            300, 300)

    # generate_warped_dataset('/home/kocur/data/BoxCars116k/dataset.pkl', '/home/kocur/data/BoxCars116k/images/',
    #                         '/home/kocur/data/BoxCars116k/dataset_warped.pkl', '/home/kocur/data/BoxCars116k/images_warped/', 320,
    #                         180)

    # generate_warped_dataset('C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/lol.pkl', 'C:/datasets/BoxCars116k/images_warped3/', 100,
    #                         100)
