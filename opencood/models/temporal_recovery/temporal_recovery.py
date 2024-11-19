import numpy as np
import torch

from opencood.data_utils.datasets import GT_RANGE
from opencood.visualization.vis_utils import plot_feature_map
from opencood.utils.box_utils import boxes_to_corners2d


def load_pkl_file(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_mask(spatial_features, corners):
    # w is the width of all objects

    # spatial features [C, H, W]
    # Create mask by H, W for all C

    # create mask for each object
    mask = np.zeros((spatial_features.shape[0], spatial_features.shape[1], spatial_features.shape[2]))

    # for all channels mask the object (same mask for all channels)
    for corner in corners:
        # get the x, y, z values of the corners
        x = corner[:, 0]
        y = corner[:, 1]

        # get the min and max values of the corners
        x_min = int(np.min(x))
        x_max = int(np.max(x))
        y_min = int(np.min(y))
        y_max = int(np.max(y))

        # create mask
        mask[:, y_min:y_max, x_min:x_max] = 1
    
    return mask


def build_lidar_spatial_feature_mask(spatial_features, object_bbx_centers):
    # GT range shape is e.g. [-W_real, -H_real, -Z_real, W_real, H_real, z_real] for lidar
    # spatial feature shape e.g. [C, H_spatial, W_spatial]
    # object bbx centers shape e.g. (N, 4, 3), the 4 corners of the 3d bounding box. (contains empty rows or max 100 vehicles)

    # ratio between spatial feature and GT range
    W_ratio = spatial_features.shape[2] / (GT_RANGE[3] - GT_RANGE[0])
    H_ratio = spatial_features.shape[1] / (GT_RANGE[4] - GT_RANGE[1])

    # delete empty object_bbx_centers
    empty_rows = np.where(np.all(object_bbx_centers == 0, axis=(1, 2)))[0]
    object_bbx_centers = np.delete(object_bbx_centers, empty_rows, axis=0)

    # object bbx centers are calculated from the center of the point cloud (add offset)
    object_bbx_centers = object_bbx_centers + np.array([GT_RANGE[3], GT_RANGE[4], GT_RANGE[5]])

    # convert x, y, z to spatial feature indices
    corner1 = object_bbx_centers[:, 0, :2] * np.array([W_ratio, H_ratio])
    corner2 = object_bbx_centers[:, 1, :2] * np.array([W_ratio, H_ratio])
    corner3 = object_bbx_centers[:, 2, :2] * np.array([W_ratio, H_ratio])
    corner4 = object_bbx_centers[:, 3, :2] * np.array([W_ratio, H_ratio])

    # corners to shape: [N, 4, 2]
    corners = np.stack([corner1, corner2, corner3, corner4], axis=1)

    mask = create_mask(spatial_features, corners)

    return mask


if __name__ == '__main__':
    object_bbx_center_pkl_path = r'/home/dominik/Git_Repos/Public/SCOPE/object_bbx_centers.pkl'
    feature_list_pkl = r'/home/dominik/Git_Repos/Public/SCOPE/feature_list.pkl'

    object_bbx_centers = load_pkl_file(object_bbx_center_pkl_path)
    # (N, 4, 3), the 4 corners of the bounding box.
    object_bbx_centers = boxes_to_corners2d(object_bbx_centers[0][0], order='hwl')
    spatial_features = load_pkl_file(feature_list_pkl)

    mask = build_lidar_spatial_feature_mask(spatial_features[0][0], object_bbx_centers)

    plot_feature_map([torch.from_numpy(mask)], save_path='my_mask.png')
    plot_feature_map([torch.from_numpy(spatial_features[0][1])], save_path='my_feature_map.png')

    # apply mask
    masked_feature_map = spatial_features[0][:] * mask
    # from [N, C, H, W] to [C, H, W]
    masked_feature_map = np.sum(masked_feature_map, axis=0)
    plot_feature_map([torch.from_numpy(masked_feature_map)], save_path='my_masked_feature_map.png')
