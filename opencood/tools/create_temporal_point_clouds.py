import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from opencood.tools import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
import pickle as pkl


def inference_temporal_potential(dataset, batch):
    gt_box_tensor, gt_object_ids = dataset.post_processor.generate_gt_bbx(batch)

    return gt_box_tensor, gt_object_ids


def main():
    HYPES_YAML_FILE = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V/config.yaml'
    TEMPORAL_STEPS = 5

    SAVE_PATH = os.path.join('visualization', 'temporal_potential', f'{TEMPORAL_STEPS}_steps')
    COMBINED_FILE_NAME = 'output.pkl'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)
    hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
    hypes['fusion']['args']['temporal_ego_only'] = False

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ALL_DATA = {}

    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_data = train_utils.to_device(batch_data, device)

        gt_box_tensor, gt_object_ids = inference_temporal_potential(opencood_dataset, batch_data[-1])

        gt_object_ids_criteria = batch_data[-1]['ego']['object_detection_info_mapping']
        gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}
        cav_object_bbx_centers = batch_data[-1]['ego']['cav_bbx_center'][-1]
        transformation_matrix = batch_data[-1]['ego']['transformation_matrix']
        cav_gt_box_tensor = opencood_dataset.post_process_cav_vehicle(cav_object_bbx_centers, transformation_matrix)

        save_path = os.path.join(SAVE_PATH, f'{i:05d}')

        pkl_element = opencood_dataset.save_temporal_point_cloud(
            None,
            gt_box_tensor,
            gt_object_ids_criteria,
            cav_gt_box_tensor,
            batch_data[-1]['ego']['origin_lidar'],
            save_path=save_path
        )

        ALL_DATA[i] = pkl_element
    
    with open(os.path.join(SAVE_PATH, COMBINED_FILE_NAME), 'wb') as f:
        pkl.dump(ALL_DATA, f)


if __name__ == '__main__':
    main()
