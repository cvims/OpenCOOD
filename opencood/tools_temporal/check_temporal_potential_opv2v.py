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


def calculate_temporal_recovered(gt_object_ids_criteria):
    # filter temporal recovered from gt_tensor
    temporal_recovered_vehicles = []
    for i, object_id in enumerate(gt_object_ids_criteria):
        if gt_object_ids_criteria[object_id]['temporal_recovered']:
            temporal_recovered_vehicles.append(i)
    
    return len(temporal_recovered_vehicles)


def main():
    HYPES_YAML_FILE = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V/config.yaml'
    TEMPORAL_STEPS = 5
    
    hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)
    hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
    hypes['fusion']['args']['temporal_ego_only'] = True

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    len_records = opencood_dataset.len_record
    scenario_temporal_recovered = dict()

    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_data = train_utils.to_device(batch_data, device)

        _, gt_object_ids = inference_temporal_potential(opencood_dataset, batch_data[-1])

        gt_object_ids_criteria = batch_data[-1]['ego']['object_detection_info_mapping']
        gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}

        for j, len_record in enumerate(len_records):
            if i <= len_record:
                if j in scenario_temporal_recovered:
                    scenario_temporal_recovered[j] += calculate_temporal_recovered(gt_object_ids_criteria)
                else:
                    scenario_temporal_recovered[j] = calculate_temporal_recovered(gt_object_ids_criteria)

                break
    
    print(scenario_temporal_recovered)
    # sum of all
    print('Total Temporal Recovered:', sum(scenario_temporal_recovered.values()))


if __name__ == '__main__':
    main()
