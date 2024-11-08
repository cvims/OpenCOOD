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
    parser.add_argument('--save_vis_path', type=str,
                        help='path to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--comm_thre', type=float, default=0,
                        help='Communication confidence threshold')
    parser.add_argument('--score_thre', type=float, default=0.23,
                    help='Confidence score threshold')
    parser.add_argument('--xyz_std', type=float, default=0.2,
                    help='position error')
    parser.add_argument('--ryp_std', type=float, default=0.2,
                help='rotation error')
    opt = parser.parse_args()
    return opt


def create_result_stat_dict():
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    return result_stat


def main():
    SAVE_VIS = False
    VIS_SAVE_PATH = os.path.join('visualization', 'inference', 'scope')

    opt = test_parser()
    opt.save_vis = SAVE_VIS
    opt.vis_save_path = VIS_SAVE_PATH
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    if opt.score_thre is not None:
        hypes['postprocess']['target_args']['score_threshold'] = opt.score_thre
    score_threshold = hypes['postprocess']['target_args']['score_threshold']
    if opt.xyz_std is not None:
        hypes['wild_setting']['xyz_std'] = opt.xyz_std
    if opt.ryp_std is not None:
        hypes['wild_setting']['ryp_std'] = opt.ryp_std
    
    # hypes['fusion']['args']['queue_length'] = 3

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=1,
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
    result_stats_levels = ['all'] #['all', 'easy', 'moderate', 'hard', 'very_hard', 'none']
    result_stats = {level: create_result_stat_dict() for level in result_stats_levels}

    temporal_result_stats = {level: create_result_stat_dict() for level in result_stats_levels}

    kitti_criteria_props = hypes['kitti_detection']['criteria']
    kitti_criteria_props['none'] = {'bbox_height': 0, 'occlusion': 1, 'truncation': 1}

    result_stats_opv2v_original = create_result_stat_dict()

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
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
        
            # Evaluate OPV2V originals (same objects as in the original dataset)
            # opv2v_original_gt_box_tensor

            gt_object_ids_criteria = batch_data[-1]['ego']['object_detection_info_mapping']
            gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}
            # batch size = 1, -> batch_data[-1]...[-1]
            cav_object_bbx_centers = batch_data[-1]['ego']['cav_bbx_center'][-1]
            transformation_matrix = batch_data[-1]['ego']['transformation_matrix']
            cav_gt_box_tensor = opencood_dataset.post_process_cav_vehicle(cav_object_bbx_centers, transformation_matrix)

            # non-temporal evaluation
            for criteria, result_stat_dict in result_stats.items():
                # without criteria
                if criteria == 'all':
                    for iou_treshold in [0.3, 0.5, 0.7]:
                        eval_utils.calculate_tp_fp(
                            pred_box_tensor,
                            pred_score,
                            gt_box_tensor,
                            result_stat_dict,
                            iou_treshold)
                else:
                    for iou_treshold in [0.3, 0.5, 0.7]:
                        camera_lidar_transform = batch_data[-1]['ego']['camera_lidar_transform']
                        # eval_utils.calculate_tp_fp_kitti(
                        #     pred_box_tensor,
                        #     pred_score,
                        #     gt_box_tensor,
                        #     result_stat_dict,
                        #     iou_treshold,
                        #     gt_object_ids_criteria,
                        #     criteria,
                        #     kitti_criteria_props[criteria],
                        #     camera_lidar_transform,
                        #     use_normal_gts=True,
                        #     use_temporal_recovered_gts=False)
                        
                        # only temporal recovered evaluation
                        eval_utils.calculate_tp_fp_kitti(
                            pred_box_tensor,
                            pred_score,
                            gt_box_tensor,
                            result_stat_dict,
                            iou_treshold,
                            gt_object_ids_criteria,
                            criteria,
                            kitti_criteria_props[criteria],
                            camera_lidar_transform,
                            use_normal_gts=True,
                            use_temporal_recovered_gts=False)
                
            # # temporal evaluation
            # for criteria, result_stat_dict in temporal_result_stats.items():
            #     # without criteria
            #     if criteria == 'all':
            #         for iou_treshold in [0.3, 0.5, 0.7]:
            #             eval_utils.calculate_tp_fp(
            #                 pred_box_tensor,
            #                 pred_score,
            #                 gt_box_tensor,
            #                 result_stat_dict,
            #                 iou_treshold)
            #     else:
            #         for iou_treshold in [0.3, 0.5, 0.7]:
            #             gt_object_ids_criteria = batch_data[-1]['ego']['object_detection_info_mapping']
            #             gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}
            #             camera_lidar_transform = batch_data[-1]['ego']['camera_lidar_transform']
            #             # eval_utils.calculate_tp_fp_kitti(
            #             #     pred_box_tensor,
            #             #     pred_score,
            #             #     gt_box_tensor,
            #             #     result_stat_dict,
            #             #     iou_treshold,
            #             #     gt_object_ids_criteria,
            #             #     criteria,
            #             #     kitti_criteria_props[criteria],
            #             #     camera_lidar_transform,
            #             #     use_normal_gts=True,
            #             #     use_temporal_recovered_gts=False)
                        
            #             # only temporal recovered evaluation
            #             eval_utils.calculate_tp_fp_kitti(
            #                 pred_box_tensor,
            #                 pred_score,
            #                 gt_box_tensor,
            #                 result_stat_dict,
            #                 iou_treshold,
            #                 gt_object_ids_criteria,
            #                 criteria,
            #                 kitti_criteria_props[criteria],
            #                 camera_lidar_transform,
            #                 use_normal_gts=False,
            #                 use_temporal_recovered_gts=True)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data[-1]['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                if opt.save_vis:
                    if not os.path.exists(opt.vis_save_path):
                        os.makedirs(opt.vis_save_path)
                    vis_save_path = os.path.join(opt.vis_save_path, '%05d.png' % i)

                # opencood_dataset.visualize_result(pred_box_tensor,
                #                                   gt_box_tensor,
                #                                   batch_data[-1]['ego'][
                #                                       'origin_lidar'],
                #                                   opt.show_vis,
                #                                   vis_save_path,
                #                                   dataset=opencood_dataset)
            
                vis_save_path = os.path.join(opt.vis_save_path, '%05d' % i)
                # saves the vis as pickle file for later use
                opencood_dataset.save_temporal_point_cloud(
                    pred_box_tensor,
                    gt_box_tensor,
                    gt_object_ids_criteria,
                    cav_gt_box_tensor,
                    batch_data[-1]['ego']['origin_lidar'],
                    save_path=vis_save_path
                )

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
    
    for criteria, result_stat_dict in temporal_result_stats.items():
        print(f'Eval results for {criteria} level:')
        eval_utils.eval_final_results(result_stat_dict,
                                      opt.model_dir,
                                      opt.global_sort_detections)
        
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
