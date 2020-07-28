import pickle
import numpy as np
import os
import cv2


# Script to generate the dataset from BoxCars116k
# for Transform2D and Transform3D approaches from the paper

def generate_ablation_dataset(dataset_path, images_path, t_dataset_path, t_images_path, im_w, im_h):
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f, encoding='latin-1', fix_imports=True)

    unacceptable = ['uvoz', 'prahaVinohradska', 'stefanikova', 'videnska']

    tds = {'samples': []}

    for s_id, sample in enumerate(ds['samples']):
        camera = sample['camera']
        print(sample['to_camera'])
        if camera in unacceptable or not sample['to_camera']:
            continue

        t_sample = {'instances': [], 'id': s_id, 'to_camera': sample['to_camera'], 'annotation': sample['annotation']}
        t_sample.update({'id': s_id})
        for instance in sample['instances']:
            path = os.path.join(images_path, instance['path'])
            image = cv2.imread(path)
            targetpath = os.path.join(t_images_path, instance['path'])
            bb3d = instance['3DBB'] - instance['3DBB_offset']

            xs = np.array([point[0] for point in bb3d])
            ys = np.array([point[1] for point in bb3d])

            xs = (im_w / float(image.shape[1])) * xs
            ys = (im_h / float(image.shape[0])) * ys

            t_image = cv2.resize(image, (im_w,im_h))

            # # remove later
            if not os.path.exists(os.path.dirname(targetpath)):
                os.makedirs(os.path.dirname(targetpath))

            print(targetpath)
            cv2.imwrite(targetpath, t_image)

            bb_out = {'x_min': np.amin(xs), 'y_min': np.amin(ys), 'x_max': np.amax(xs), 'y_max': np.amax(ys)}

            t_instance = {'filename': instance['path'], 'bb_out': bb_out,
                          'instance_id': instance['instance_id']}
            t_sample['instances'].append(t_instance)
        tds['samples'].append(t_sample)

    with open(t_dataset_path, "wb") as f:
        pickle.dump(tds, f)


if __name__ == "__main__":
    # generate_warped_dataset('C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/dataset_warped.pkl', 'C:/datasets/BoxCars116k/images_warped/', 300,
    #                         300)

    # generate_ablation_dataset('C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/dataset_ablation.pkl', 'C:/datasets/BoxCars116k/images_ablation/', 320,
    #                         180)

    generate_ablation_dataset('/home/k/kocur15/data/BoxCars116k/dataset.pkl', '/home/k/kocur15/data/BoxCars116k/images/',
                            '/home/k/kocur15/data/BoxCars116k/dataset_ablation.pkl', '/home/k/kocur15/data/BoxCars116k/images_ablation/', 320, 180)

    # generate_warped_dataset('C:/datasets/BoxCars116k/dataset.pkl', 'C:/datasets/BoxCars116k/images/',
    #                         'C:/datasets/BoxCars116k/lol.pkl', 'C:/datasets/BoxCars116k/images_warped2/', 100,
    #                         100)
