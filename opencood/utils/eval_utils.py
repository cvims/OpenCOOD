# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import cv2
import numpy as np
import torch

from opencood.utils import common_utils, box_utils, camera_utils, temporal_utils, transformation_utils
from opencood.hypes_yaml import yaml_utils


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


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
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


def caluclate_tp_fp_kitti(
        det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
        gt_object_ids_criteria, criteria, criteria_props, camera_lidar_transform,
        org_image_width=800, org_image_height=600):
    bbox_height_criteria = criteria_props['bbox_height']
    occlusion_criteria = criteria_props['occlusion']
    truncation_criteria = criteria_props['truncation']
    # https://github.com/traveller59/kitti-object-eval-python/blob/master/eval.py

    criteria_id = temporal_utils.KITTI_DETECTION_CATEGORY_ENUM[criteria]

    # index of gt_object_ids_criteria (with is a dict of key: object_id, value: dict)
    criteria_considered_gts = set()
    criteria_considered_gt_criteria = dict()
    for i, object_id in enumerate(gt_object_ids_criteria):
        kitti_criteria = gt_object_ids_criteria[object_id]['kitti_criteria']
        if kitti_criteria <= criteria_id:
            criteria_considered_gts.add(i)
            criteria_considered_gt_criteria[i] = gt_object_ids_criteria[object_id]

    # set with index of det_boxes as key
    criteria_considered_dets = set()

    for camera_transform in camera_lidar_transform:
        for camera in camera_transform:
            camera_extrinsics = camera_transform[camera]['camera_extrinsic_to_ego_lidar'].cpu().numpy()
            camera_intrinsics = camera_transform[camera]['camera_intrinsic'].cpu().numpy()

            print('TODO: Bilder von CAVs haben falsche Projektion, immer noch?')

            # inverse camera extrinsics
            camera_extrinsics = np.linalg.inv(camera_extrinsics)

            cam_det_boxes = camera_utils.project_3d_to_camera_torch(det_boxes, camera_intrinsics, camera_extrinsics)
            # load image
            image = camera_transform[camera]['image_path']
            image = cv2.imread(image)
            image, filtered_cam_boxes, filter_mask = camera_utils.draw_2d_bbx(image, cam_det_boxes.cpu().numpy())
            # save image
            cv2.imwrite('test.jpg', image)

            _, filter_mask = camera_utils.filter_bbx_out_scope_torch(cam_det_boxes, org_image_width, org_image_height)

            # max - min of :,:,1 (keep dims of filtered_cam_boxes)
            bbox_heights = torch.max(cam_det_boxes[:, :, 1], dim=1)[0] - torch.min(cam_det_boxes[:, :, 1], dim=1)[0]

            min_bbox_height_filter_mask = bbox_heights >= bbox_height_criteria
            # combine masks
            filter_mask = torch.logical_and(filter_mask, min_bbox_height_filter_mask)

            # save indices of values that are True
            criteria_considered_dets.update(torch.nonzero(filter_mask).flatten().tolist())

            # same for gt_boxes
            cam_gt_boxes = camera_utils.project_3d_to_camera_torch(gt_boxes, camera_intrinsics, camera_extrinsics)

            _, filter_mask = camera_utils.filter_bbx_out_scope_torch(cam_gt_boxes, org_image_width, org_image_height)

            # max - min of :,:,1 (keep dims of filtered_cam_boxes)
            bbox_heights = torch.max(cam_gt_boxes[:, :, 1], dim=1)[0] - torch.min(cam_gt_boxes[:, :, 1], dim=1)[0]

            min_bbox_height_filter_mask = bbox_heights >= bbox_height_criteria
            # combine masks
            filter_mask = torch.logical_and(filter_mask, min_bbox_height_filter_mask)

            # save indices of values that are True
            criteria_considered_gts.update(torch.nonzero(filter_mask).flatten().tolist())
    
    considered_dets = det_boxes[list(criteria_considered_dets)]
    considered_det_scores = det_score[list(criteria_considered_dets)]

    considered_gts = gt_boxes[list(criteria_considered_gts)]

    return caluclate_tp_fp(considered_dets, considered_det_scores, considered_gts, result_stat, iou_thresh)

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


def eval_final_results(result_stat, save_path, global_sort_detections):
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
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print('The Average Precision at IOU 0.3 is %.2f, '
          'The Average Precision at IOU 0.5 is %.2f, '
          'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))
