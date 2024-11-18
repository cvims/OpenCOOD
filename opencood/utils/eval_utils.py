# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from opencood.utils import common_utils, box_utils, camera_utils, temporal_utils, transformation_utils
from opencood.hypes_yaml import yaml_utils


def set_random_seed(seed):
    """
    Set the random seed.

    Parameters
    ----------
    seed : int
        The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def calculate_temporal_recovered_hits(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, gt_object_ids_criteria):
    # match detection and groundtruth bounding box by Hungarian algorithm
    matched_pairs_indices = match_gt_det_hungarian(det_boxes, gt_boxes, iou_thresh)
    # unmatched_detections_indices = set(range(det_boxes.shape[0])) - set([pair[0] for pair in matched_pairs_indices])
    unmatched_gt_indices = set(range(gt_boxes.shape[0])) - set([pair[1] for pair in matched_pairs_indices])

    filter_gts_idx = {i for i, gt_idx in enumerate(gt_object_ids_criteria) if gt_object_ids_criteria[gt_idx]['temporal_recovered']}

    if len(filter_gts_idx) == 0:
        return
    
    matched_pairs_indices = [(det_idx, gt_idx) for det_idx, gt_idx in matched_pairs_indices if gt_idx in filter_gts_idx]
    unmatched_gt_indices = {gt_idx for gt_idx in unmatched_gt_indices if gt_idx in filter_gts_idx}

    hits = len(matched_pairs_indices)
    no_hits = len(unmatched_gt_indices)

    result_stat[iou_thresh]['hits'] += hits
    result_stat[iou_thresh]['no_hits'] += no_hits


def calculate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def match_gt_det_hungarian(det_boxes, gt_boxes, iou_thresh):
    """
    Match the detection bounding box with the groundtruth bounding box by
    Hungarian algorithm.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3).
    gt_boxes : torch.Tensor
        The groundtruth bounding box, shape (M, 8, 3).
    iou_thresh : float
        The iou thresh.

    Returns
    -------
    matched_pairs : list
        The matched pairs of detection and groundtruth bounding box by hungarian algorithm.
    """
    # Step 1: Calculate IoU between each detection and ground truth box
    iou_matrix = calculate_iou(det_boxes, gt_boxes)
    
    # Step 2: Create the cost matrix (1 - IoU) for the Hungarian algorithm
    cost_matrix = 1 - iou_matrix.cpu().numpy()  # convert to numpy for linear_sum_assignment

    # Step 3: Apply Hungarian Algorithm (linear sum assignment)
    det_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Step 4: Filter matched pairs by IoU threshold
    matched_pairs = []
    for det_idx, gt_idx in zip(det_indices, gt_indices):
        if iou_matrix[det_idx, gt_idx] >= iou_thresh:
            matched_pairs.append((det_idx, gt_idx))
    
    return matched_pairs


def calculate_iou(det_boxes, gt_boxes):
    """
    Calculate the IoU between detection and groundtruth bounding box.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3).
    gt_boxes : torch.Tensor
        The groundtruth bounding box, shape (M, 8, 3).
    
    Returns
    -------
    iou_matrix : torch.Tensor
        The IoU matrix, shape (N, M).
    """
    if det_boxes is None:
        return torch.zeros((0, 0))

    if gt_boxes is None:
        return torch.zeros((det_boxes.shape[0], 0))

    # convert bounding boxes to numpy array
    det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
    gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

    # convert bounding boxes to polygon format
    det_polygon_list = list(common_utils.convert_format(det_boxes))
    gt_polygon_list = list(common_utils.convert_format(gt_boxes))

    # calculate IoU between each detection and ground truth box
    iou_matrix = np.zeros((len(det_polygon_list), len(gt_polygon_list)))
    for i in range(len(det_polygon_list)):
        iou_matrix[i, :] = common_utils.compute_iou(det_polygon_list[i], gt_polygon_list)

    return torch.tensor(iou_matrix)


def calculate_tp_fp_kitti(
        det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
        gt_object_ids_criteria, criteria, criteria_props, camera_lidar_transform,
        use_normal_gts=True, use_temporal_recovered_gts=False, use_temporal_kitti_criteria=False,
        org_image_width=800, org_image_height=600):

    # match detection and groundtruth bounding box by Hungarian algorithm
    matched_pairs_indices = match_gt_det_hungarian(det_boxes, gt_boxes, iou_thresh)
    unmatched_detections_indices = set(range(det_boxes.shape[0])) - set([pair[0] for pair in matched_pairs_indices])

    filter_gts_idx = set()
    if use_normal_gts and use_temporal_recovered_gts:
        pass
    elif use_temporal_recovered_gts:
        # search for gt_object_ids_criteria with property temporal_recovered
        filter_gts_idx = {i for i, gt_idx in enumerate(gt_object_ids_criteria) if gt_object_ids_criteria[gt_idx]['temporal_recovered']}
        # if filter_gt_ids is empty, set it to -1
        if len(filter_gts_idx) == 0:
            # if there are no temporal recovered gts, return
            filter_gts_idx = set([-1])
    elif use_normal_gts:
        filter_gts_idx = {i for i, gt_idx in enumerate(gt_object_ids_criteria) if not gt_object_ids_criteria[gt_idx]['temporal_recovered']}
    else:
        raise Exception('Either use_normal_gts or use_temporal_recovered_gts must be True')
    
    if len(filter_gts_idx) == 1 and -1 in filter_gts_idx:
        # nothing to do
        return
    
    if len(filter_gts_idx) > 0:
        # second dim of matched_pairs_indices is the index of gt_boxes
        # filter out the matched pairs that are not temporal recovered
        matched_pairs_indices = [(det_idx, gt_idx) for det_idx, gt_idx in matched_pairs_indices if gt_idx in filter_gts_idx]

        # # get det_boxes, det_score, and gt_boxes with indices
        # det_boxes = det_boxes[[pair[0] for pair in matched_pairs_indices]]
        # det_score = det_score[[pair[0] for pair in matched_pairs_indices]]
        # gt_boxes = gt_boxes[[pair[1] for pair in matched_pairs_indices]]

        # filter_gts_idx to actual key
        gts_keys = list(gt_object_ids_criteria.keys())
        gts_keys = {gts_keys[i] for i in filter_gts_idx}
        gt_object_ids_criteria = {gt_idx: gt_object_ids_criteria[gt_idx] for gt_idx in gts_keys}

    bbox_height_criteria = criteria_props['bbox_height']
    # occlusion_criteria = criteria_props['occlusion']
    # truncation_criteria = criteria_props['truncation']

    criteria_id = temporal_utils.KITTI_DETECTION_CATEGORY_ENUM[criteria]

    # index of gt_object_ids_criteria (with is a dict of key: object_id, value: dict)
    criteria_considered_gts = set()
    criteria_considered_gt_criteria = dict()
    if use_temporal_kitti_criteria:
        attr_key = 'temporal_kitti_criteria'
    else:
        attr_key = 'kitti_criteria'

    for i, object_id in enumerate(gt_object_ids_criteria):
        kitti_criteria = gt_object_ids_criteria[object_id][attr_key]
        if kitti_criteria <= criteria_id:
            criteria_considered_gts.add(i)
            criteria_considered_gt_criteria[i] = gt_object_ids_criteria[object_id]
    
    if len(criteria_considered_gts) == 0:
        return
    
    # remove matched pairs that are not considered based on criteria
    matched_pairs_indices = [(det_idx, gt_idx) for det_idx, gt_idx in matched_pairs_indices if gt_idx in criteria_considered_gts]

    matched_det_boxes = det_boxes[[pair[0] for pair in matched_pairs_indices]]
    matched_det_scores = det_score[[pair[0] for pair in matched_pairs_indices]]
    unmatched_det_boxes = det_boxes[list(unmatched_detections_indices)]
    unmatched_det_scores = det_score[list(unmatched_detections_indices)]

    matched_gt_boxes = gt_boxes[[pair[1] for pair in matched_pairs_indices]]

    # For all unmatched det boxes, check bounding box height (in camera coordinates)
    # So that only the detections with the the bounding box height of the chosen criteria are considered
    unmatched_considered_dets_indices = set()
    for camera_transform in camera_lidar_transform:
        for camera in camera_transform:
            camera_extrinsics = camera_transform[camera]['camera_extrinsic_to_ego_lidar'].cpu().numpy()
            camera_intrinsics = camera_transform[camera]['camera_intrinsic'].cpu().numpy()

            # inverse camera extrinsics
            camera_extrinsics = np.linalg.inv(camera_extrinsics)

            unmatched_cam_det_boxes = camera_utils.project_3d_to_camera_torch(unmatched_det_boxes, camera_intrinsics, camera_extrinsics)
            # # load image
            # image = camera_transform[camera]['image_path']
            # image = cv2.imread(image)
            # image, filtered_cam_boxes, filter_mask = camera_utils.draw_2d_bbx(image, cam_det_boxes.cpu().numpy())
            # # save image
            # cv2.imwrite('test.jpg', image)

            _, filter_mask = camera_utils.filter_bbx_out_scope_torch(unmatched_cam_det_boxes, org_image_width, org_image_height)

            # max - min of :,:,1 (keep dims of filtered_cam_boxes)
            bbox_heights = torch.max(unmatched_cam_det_boxes[:, :, 1], dim=1)[0] - torch.min(unmatched_cam_det_boxes[:, :, 1], dim=1)[0]

            min_bbox_height_filter_mask = bbox_heights >= bbox_height_criteria
            # combine masks
            filter_mask = torch.logical_and(filter_mask, min_bbox_height_filter_mask)

            # save indices of values that are True
            unmatched_considered_dets_indices.update(torch.nonzero(filter_mask).flatten().tolist())

    unmatched_considered_dets_indices = list(unmatched_considered_dets_indices)
    if unmatched_considered_dets_indices:
        unmatched_considered_dets = unmatched_det_boxes[unmatched_considered_dets_indices]
        unmatched_considered_det_scores = unmatched_det_scores[unmatched_considered_dets_indices]

        # combine matched and unmatched detections
        considered_dets = torch.cat([matched_det_boxes, unmatched_considered_dets], dim=0)
        considered_det_scores = torch.cat([matched_det_scores, unmatched_considered_det_scores], dim=0)
    else:
        considered_dets = matched_det_boxes
        considered_det_scores = matched_det_scores

    considered_gts = gt_boxes[list(criteria_considered_gts)]

    calculate_tp_fp(considered_dets, considered_det_scores, considered_gts, result_stat, iou_thresh)


def calculate_tp_fp_kitti_temporal_recovered(
        det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
        gt_object_ids_criteria, criteria, criteria_props, camera_lidar_transform,
        org_image_width=800, org_image_height=600):
    
    matched_pairs_indices = match_gt_det_hungarian(det_boxes, gt_boxes, iou_thresh)
    # search for gt_object_ids_criteria with property temporal_recovered
    temporal_recovered_ids = {i for i, gt_idx in enumerate(gt_object_ids_criteria) if gt_object_ids_criteria[gt_idx]['temporal_recovered']}

    if not temporal_recovered_ids:
        return

    # second dim of matched_pairs_indices is the index of gt_boxes
    # filter out the matched pairs that are not temporal recovered
    matched_pairs_indices = [(det_idx, gt_idx) for det_idx, gt_idx in matched_pairs_indices if gt_idx in temporal_recovered_ids]

    # get det_boxes, det_score, and gt_boxes with indices
    matched_det_boxes = det_boxes[[pair[0] for pair in matched_pairs_indices]]
    matched_det_scores = det_score[[pair[0] for pair in matched_pairs_indices]]
    matched_gt_boxes = gt_boxes[[pair[1] for pair in matched_pairs_indices]]

    temporal_recovered_gt_object_ids_criteria = {gt_idx: gt_object_ids_criteria[gt_idx] for gt_idx in temporal_recovered_ids}

    return calculate_tp_fp_kitti(
        matched_det_boxes, matched_det_scores, matched_gt_boxes, result_stat, iou_thresh,
        temporal_recovered_gt_object_ids_criteria, criteria, criteria_props, camera_lidar_transform,
        org_image_width, org_image_height)                                 


# def caluclate_tp_fp_kitti(
#         det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
#         gt_object_ids_criteria, criteria, criteria_props, camera_lidar_transform,
#         org_image_width=800, org_image_height=600):

#     bbox_height_criteria = criteria_props['bbox_height']
#     occlusion_criteria = criteria_props['occlusion']
#     truncation_criteria = criteria_props['truncation']
#     # https://github.com/traveller59/kitti-object-eval-python/blob/master/eval.py

#     criteria_id = temporal_utils.KITTI_DETECTION_CATEGORY_ENUM[criteria]

#     # index of gt_object_ids_criteria (with is a dict of key: object_id, value: dict)
#     criteria_considered_gts = set()
#     criteria_considered_gt_criteria = dict()
#     for i, object_id in enumerate(gt_object_ids_criteria):
#         kitti_criteria = gt_object_ids_criteria[object_id]['kitti_criteria']
#         if kitti_criteria <= criteria_id:
#             criteria_considered_gts.add(i)
#             criteria_considered_gt_criteria[i] = gt_object_ids_criteria[object_id]

#     # set with index of det_boxes as key
#     criteria_considered_dets = set()

#     for camera_transform in camera_lidar_transform:
#         for camera in camera_transform:
#             camera_extrinsics = camera_transform[camera]['camera_extrinsic_to_ego_lidar'].cpu().numpy()
#             camera_intrinsics = camera_transform[camera]['camera_intrinsic'].cpu().numpy()

#             # inverse camera extrinsics
#             camera_extrinsics = np.linalg.inv(camera_extrinsics)

#             cam_det_boxes = camera_utils.project_3d_to_camera_torch(det_boxes, camera_intrinsics, camera_extrinsics)
#             # # load image
#             # image = camera_transform[camera]['image_path']
#             # image = cv2.imread(image)
#             # image, filtered_cam_boxes, filter_mask = camera_utils.draw_2d_bbx(image, cam_det_boxes.cpu().numpy())
#             # # save image
#             # cv2.imwrite('test.jpg', image)

#             _, filter_mask = camera_utils.filter_bbx_out_scope_torch(cam_det_boxes, org_image_width, org_image_height)

#             # max - min of :,:,1 (keep dims of filtered_cam_boxes)
#             bbox_heights = torch.max(cam_det_boxes[:, :, 1], dim=1)[0] - torch.min(cam_det_boxes[:, :, 1], dim=1)[0]

#             min_bbox_height_filter_mask = bbox_heights >= bbox_height_criteria
#             # combine masks
#             filter_mask = torch.logical_and(filter_mask, min_bbox_height_filter_mask)

#             # save indices of values that are True
#             criteria_considered_dets.update(torch.nonzero(filter_mask).flatten().tolist())

#             # same for gt_boxes
#             cam_gt_boxes = camera_utils.project_3d_to_camera_torch(gt_boxes, camera_intrinsics, camera_extrinsics)

#             _, filter_mask = camera_utils.filter_bbx_out_scope_torch(cam_gt_boxes, org_image_width, org_image_height)

#             # max - min of :,:,1 (keep dims of filtered_cam_boxes)
#             bbox_heights = torch.max(cam_gt_boxes[:, :, 1], dim=1)[0] - torch.min(cam_gt_boxes[:, :, 1], dim=1)[0]

#             min_bbox_height_filter_mask = bbox_heights >= bbox_height_criteria
#             # combine masks
#             filter_mask = torch.logical_and(filter_mask, min_bbox_height_filter_mask)

#             # save indices of values that are True
#             criteria_considered_gts.update(torch.nonzero(filter_mask).flatten().tolist())
    
#     considered_dets = det_boxes[list(criteria_considered_dets)]
#     considered_det_scores = det_score[list(criteria_considered_dets)]

#     considered_gts = gt_boxes[list(criteria_considered_gts)]

#     return caluclate_tp_fp(considered_dets, considered_det_scores, considered_gts, result_stat, iou_thresh)

def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path=None, global_sort_detections=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    
    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'

    if save_path:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print('The Average Precision at IOU 0.3 is %.2f, '
          'The Average Precision at IOU 0.5 is %.2f, '
          'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))
