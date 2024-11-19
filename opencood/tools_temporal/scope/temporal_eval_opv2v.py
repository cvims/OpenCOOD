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
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    return result_stat


def create_temporal_result_stat_dict():
    result_stat = {0.3: {'hits': 0, 'no_hits': 0},
                   0.5: {'hits': 0, 'no_hits': 0},
                   0.7: {'hits': 0, 'no_hits': 0}}
    return result_stat


def main():
    eval_utils.set_random_seed(0)

    DATA_PATH = '/data/public_datasets/OPV2V/original/train'

    MODEL_DIRS = [
        r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V',
        # r'/home/dominik/Git_Repos/Private/OpenCOOD/runs/scope/point_pillar_scope_more_steps_all_cavs_2024_11_12_19_54_32',
        # r'/home/dominik/Git_Repos/Private/OpenCOOD/runs/temporal/scope/202411131117',
        # r'/home/dominik/Git_Repos/Private/OpenCOOD/runs/temporal/scope/scope_temporal_4_steps_2024_11_15_11_50_15'
    ]

    HYPES_YAML_FILES = [os.path.join(model_dir, 'config.yaml') for model_dir in MODEL_DIRS]

    # STANDARD SCOPE SETTING: Temporal steps = 2; Temporal ego only: True

    TEMPORAL_STEPS = 4
    TEMPORAL_EGO_ONLY = False
    
    print(f'Eval for all models with configs: temporal steps = {TEMPORAL_STEPS}, temporal ego only = {TEMPORAL_EGO_ONLY}')

    for MODEL_DIR, HYPES_YAML_FILE in zip(MODEL_DIRS, HYPES_YAML_FILES):
        hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)

        hypes['root_dir'] = DATA_PATH
        hypes['validate_dir'] = DATA_PATH

        hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
        hypes['fusion']['args']['temporal_ego_only'] = TEMPORAL_EGO_ONLY

        hypes['model']['args']['fusion_args']['communication']['thre'] = 0
        hypes['postprocess']['target_args']['score_threshold'] = 0.23
        hypes['wild_setting']['xyz_std'] = 0.2
        hypes['wild_setting']['ryp_std'] = 0.2

        hypes['train_params']['batch_size'] = 1
        hypes['train_params']['frame'] = TEMPORAL_STEPS - 1
        hypes['model']['args']['fusion_args']['frame'] = TEMPORAL_STEPS - 1

        # DEBUG
        use_scenarios_idx = [0]
        use_scenarios_idx = None

        print('Dataset Building')
        opencood_dataset = build_dataset(
            hypes, visualize=True, train=False,
            use_scenarios_idx=use_scenarios_idx,
            preload_lidar_files=True
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

        standard_result_stats = create_result_stat_dict()
        temporal_result_stats = create_temporal_result_stat_dict()

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

                # standard evaluation
                for iou_thre in [0.3, 0.5, 0.7]:
                    eval_utils.calculate_tp_fp(
                        pred_box_tensor,
                        pred_score,
                        gt_box_tensor,
                        standard_result_stats,
                        iou_thre)

                # temporal evaluation
                for iou_thre in [0.3, 0.5, 0.7]:
                    eval_utils.calculate_temporal_recovered_hits(
                        pred_box_tensor,
                        pred_score,
                        gt_box_tensor,
                        temporal_result_stats,
                        iou_thre,
                        gt_object_ids_criteria)
        

        print(f'Results for {MODEL_DIR}')
        eval_utils.eval_final_results(standard_result_stats, None, None)
        print(f'Temporal Evaluation Results:', temporal_result_stats)

if __name__ == '__main__':
    main()
