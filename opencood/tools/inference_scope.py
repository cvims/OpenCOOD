# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        default=r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V',
                        required=False,
                        help='Continued training path')
    parser.add_argument('--fusion_method',
                        required=False, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def create_result_stat_dict():
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    return result_stat


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
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
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stats_levels = ['all', 'easy'] #, 'moderate', 'hard', 'very_hard']
    result_stats = {level: create_result_stat_dict() for level in result_stats_levels}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, frame_object_visibility_mapping, temporal_object_visibility_mapping = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset,
                                                          return_object_criteria=True)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, frame_object_visibility_mapping, temporal_object_visibility_mapping = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset,
                                                           return_object_criteria=True)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, frame_object_visibility_mapping, temporal_object_visibility_mapping = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  return_object_criteria=True)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            for criteria, result_stat_dict in result_stats.items():
                # filter prediction and gt bounding box by difficulty
                criteria_level = 0
                if criteria == 'easy':
                    criteria_level = 0
                elif criteria == 'moderate':
                    criteria_level = 1
                elif criteria == 'hard':
                    criteria_level = 2
                elif criteria == 'very_hard':
                    criteria_level = 3
                else:
                    criteria_level = -1  # all
                
                # filter the gt_box by difficulty
                if criteria_level == -1:
                    filtered_gt_box_tensor = gt_box_tensor
                else:
                    selected_ids = []
                    selected_object_ids = set()
                    for i, sel_id in enumerate(temporal_object_visibility_mapping):
                        if temporal_object_visibility_mapping[sel_id] == criteria_level:
                            selected_ids.append(i)
                            selected_object_ids.add(sel_id)
                    
                    filtered_gt_box_tensor = gt_box_tensor[selected_ids]
                
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    filtered_gt_box_tensor,
                    result_stat_dict,
                    0.3)
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    filtered_gt_box_tensor,
                    result_stat_dict,
                    0.5)
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    filtered_gt_box_tensor,
                    result_stat_dict,
                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    for criteria, result_stat_dict in result_stats.items():
        print(f'Eval results for {criteria} level:')
        eval_utils.eval_final_results(result_stat_dict,
                                      opt.model_dir,
                                      opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()