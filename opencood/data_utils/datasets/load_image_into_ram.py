from opencood.utils.camera_utils import load_rgb_from_files
from opencood.hypes_yaml.yaml_utils import load_yaml
import pickle
import os
import cv2
import tqdm


def load_images_into_ram(root_dir, all_yamls):
    all_images = {}

    for folder in tqdm.tqdm(all_yamls):
        all_images[folder] = {}
        full_path = os.path.join(root_dir, folder)
        for cav_id in all_yamls[folder]:
            all_images[folder][cav_id] = {}
            for timestamp in all_yamls[folder][cav_id]:
                cam0_pth = os.path.join(full_path, str(cav_id), f'{timestamp}_camera0.png')
                cam1_pth = os.path.join(full_path, str(cav_id), f'{timestamp}_camera1.png')
                cam2_pth = os.path.join(full_path, str(cav_id), f'{timestamp}_camera2.png')
                cam3_pth = os.path.join(full_path, str(cav_id), f'{timestamp}_camera3.png')

                # load them as numpy array
                cam0 = cv2.imread(cam0_pth)
                cam1 = cv2.imread(cam1_pth)
                cam2 = cv2.imread(cam2_pth)
                cam3 = cv2.imread(cam3_pth)

                all_images[folder][cav_id][timestamp] = {
                    'camera0': cam0,
                    'camera1': cam1,
                    'camera2': cam2,
                    'camera3': cam3
                }
                
    print('test')

if __name__ == '__main__':
    root_dir = r'/data/public_datasets/OPV2V/original/train'
    all_yamls = pickle.load(open(os.path.join(root_dir, 'yamls.pkl'), 'rb'))

    load_images_into_ram(root_dir, all_yamls)

