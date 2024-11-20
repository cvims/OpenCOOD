import torch
import torch.nn as nn
from opencood.data_utils.datasets import GT_RANGE
from opencood.utils.box_utils import boxes_to_corners2d

from opencood.visualization.vis_utils import plot_feature_map


def create_temporal_gt_mask(spatial_features, object_bbx_centers, gt_mapping_info):
    """
    Create temporal ground truth mask for the temporal mask model
    :param spatial_features: spatial features from the scope model
        Shape: [B, H, W]
    :param object_bbx_centers: object bounding box centers from the scope model
        Shape: [B, MAX_PREDICTIONS, 7]
    """
    # delete empty object_bbx_centers
    non_empty_rows = []
    for object_bbx_center in object_bbx_centers:
        empty_row = torch.where(torch.all(object_bbx_center != 0, dim=1))[0]
        non_empty_rows.append(empty_row)

    object_bbx_centers = [object_bbx_centers[i][non_empty_rows[i]] for i in range(len(object_bbx_centers))]

    # filter only temporal vehicles
    temporal_indices = []
    for i, mapping_info in enumerate(gt_mapping_info):
        temporal_vehicle_indices = [i for i, o_id in enumerate(mapping_info) if mapping_info[o_id]['temporal_recovered']]
        temporal_indices.append(temporal_vehicle_indices)
    
    # get temporals in object_bbx_centers
    for i, temporal_vehicle_indices in enumerate(temporal_indices):
        object_bbx_centers[i] = object_bbx_centers[i][temporal_vehicle_indices]

    # centers to corners
    corners_list = [boxes_to_corners2d(object_bbx_center, order='hwl') for object_bbx_center in object_bbx_centers]

    # object bbx centers are calculated from the center of the point cloud (add offset)
    for i, corners in enumerate(corners_list):
        corners_list[i] = corners + torch.tensor([GT_RANGE[3], GT_RANGE[4], GT_RANGE[5]]).to(corners.device)

    # Calculate ratio between spatial feature and GT range
    W_ratio = spatial_features.shape[2] / (GT_RANGE[3] - GT_RANGE[0])
    H_ratio = spatial_features.shape[1] / (GT_RANGE[4] - GT_RANGE[1])

    # convert x, y, z of corners to spatial feature indices
    for i, corners in enumerate(corners_list):
        corners_list[i] = corners[:, :, :2] * torch.tensor([W_ratio, H_ratio]).to(corners.device)

    # create mask for each object
    mask = torch.zeros((spatial_features.shape[0], spatial_features.shape[1], spatial_features.shape[2]))

    for i, corners in enumerate(corners_list):
        for corner in corners:
            x = corner[:, 0]
            y = corner[:, 1]

            x_min = int(torch.min(x))
            x_max = int(torch.max(x))
            y_min = int(torch.min(y))
            y_max = int(torch.max(y))

            mask[i, y_min:y_max, x_min:x_max] = 1

    return mask.to(device=spatial_features.device)


class TemporalMaskBCELoss(nn.Module):
    def __init__(self, args, pos_weight=50.0, neg_weight=1.0):
        super(TemporalMaskBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        # self.loss = nn.BCELoss()
        self.loss_dict = {}
    
    def weighted_bce_loss(self, mask_pred, temporal_gt_mask):
        """
        Calculate the weighted binary cross entropy loss
        :param mask_pred: predicted mask
        :param temporal_gt_mask: temporal ground truth mask
        :param pos_weight: positive weight
        :param neg_weight: negative weight
        """
        loss = -self.pos_weight * temporal_gt_mask * torch.log(mask_pred + 1e-12) - self.neg_weight * (1 - temporal_gt_mask) * torch.log(1 - mask_pred + 1e-12)
        return loss.mean()

    def forward(self, mask_pred, object_bbx_centers, gt_mapping_info):
        temporal_gt_mask = create_temporal_gt_mask(mask_pred, object_bbx_centers, gt_mapping_info)

        final_loss = self.weighted_bce_loss(mask_pred, temporal_gt_mask)

        self.loss_dict.update(
            {
                'total_loss': final_loss
            }
        )

        return final_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total_loss = self.loss_dict['total_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item()))

        if writer:
            writer.add_scalar('Total_loss', total_loss.item(),
                            epoch*batch_len + batch_id)
