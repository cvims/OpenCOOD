# How many objects does scope recover from historical frames?

# 1. Eval temporal vehicles only
# 2. Count hits and missings based on IoU

import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from opencood.tools import train_utils, inference_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def create_result_stat_dict():
    result_stat = {0.3: {'hits': 0, 'no_hits': 0},
                   0.5: {'hits': 0, 'no_hits': 0},
                   0.7: {'hits': 0, 'no_hits': 0}}
    return result_stat


def main():
    MODEL_DIR = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V'
    HYPES_YAML_FILE = os.path.join(MODEL_DIR, 'config.yaml')
    TEMPORAL_STEPS = 2
    
    hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)
    hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
    hypes['fusion']['args']['temporal_ego_only'] = True

    hypes['model']['args']['fusion_args']['communication']['thre'] = 0
    hypes['postprocess']['target_args']['score_threshold'] = 0.23
    hypes['wild_setting']['xyz_std'] = 0.2
    hypes['wild_setting']['ryp_std'] = 0.2

    # DEBUG
    use_scenarios_idx = [0]

    print('Dataset Building')
    opencood_dataset = build_dataset(
        hypes, visualize=True, train=False,
        use_scenarios_idx=use_scenarios_idx,
    )
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(MODEL_DIR, model)
    model.eval()

    temporal_result_stats = create_result_stat_dict()

    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
        
            gt_object_ids_criteria = batch_data[-1]['ego']['object_detection_info_mapping']
            gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}

            for iou_thre in [0.3, 0.5, 0.7]:
                eval_utils.calculate_temporal_recovered_hits(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    temporal_result_stats,
                    iou_thre,
                    gt_object_ids_criteria)
    
    print(temporal_result_stats)


if __name__ == '__main__':
    main()
