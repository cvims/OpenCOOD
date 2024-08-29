import opencood.hypes_yaml.yaml_utils as yaml_utils
import open3d as o3d
import os
import numpy as np


def load_pcd_file(pcd_file):
    """
    Load a pcd file
    :param pcd_file: path to the pcd file
    :return: point cloud
    """
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd


def draw_3d_bbox(pcd, v_3d_boxes):
    """
    Draw 3d bounding boxes into the point cloud
    :param pcd: point cloud
    :param v_3d_boxes: list of 3d bounding boxes
    :return: point cloud with 3d bounding boxes
    """
    combined_pcd = pcd

    for v_3d_box in v_3d_boxes:
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        colors = [[1, 0, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(v_3d_box),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # Sample points from the lines and add them to the point cloud
        sampled_points = []
        num_points_per_line = 10

        for line in lines:
            for i in range(num_points_per_line):
                point = (v_3d_box[line[0]] * (num_points_per_line - i) + v_3d_box[line[1]] * i) / num_points_per_line
                sampled_points.append(point)
        
        sampled_points = np.asarray(sampled_points)
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd.paint_uniform_color([0, 1, 0])

        combined_pcd += sampled_pcd

    return combined_pcd


def calculate_lidar_hits(pcd, lidar_pose, vehicles):
    # x,y,z
    lidar_position = np.asarray(lidar_pose[:3]) # world coordinates (x,y,z)
    lidar_rotation = np.asarray(lidar_pose[3:]) # pitch, yaw, roll

    # calculate the world coordinates of the pcd points (consider rotation and translation)
    pcd_points = np.asarray(pcd.points)
    lidar_yaw = np.deg2rad(lidar_rotation[1])
    R = np.array([
        [np.cos(lidar_yaw), -np.sin(lidar_yaw), 0],
        [np.sin(lidar_yaw), np.cos(lidar_yaw), 0],
        [0, 0, 1]
    ])

    pcd_points = np.dot(R, pcd_points.T).T
    pcd_points += lidar_position

    v_3d_boxes = []
    for v_id, v_data in vehicles.items():
        angle = v_data['angle'] # pitch, yaw, roll
        extent = v_data['extent']
        location = v_data['location'] # world coordinates

        v_3d_bbox = generate_3d_bbox(extent, location, angle)
        # add lidar position to the 3d bounding box
        v_3d_bbox -= lidar_position
        v_3d_boxes.append(v_3d_bbox)
    
    # draw the 3d bounding boxes into point cloud
    pcd = draw_3d_bbox(pcd, v_3d_boxes)

    # save pcd
    o3d.io.write_point_cloud('output.pcd', pcd)


def generate_3d_bbox(extent, location, angle):
    bbox_3d = np.array([
        [-extent[0], -extent[1], 0],
        [extent[0], -extent[1], 0],
        [extent[0], extent[1], 0],
        [-extent[0], extent[1], 0],
        [-extent[0], -extent[1], extent[2]*2],
        [extent[0], -extent[1], extent[2]*2],
        [extent[0], extent[1], extent[2]*2],
        [-extent[0], extent[1], extent[2]*2]
    ])

    yaw_rad = np.deg2rad(angle[1])
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    bbox_3d = np.dot(R, bbox_3d.T).T
    bbox_3d += location

    return bbox_3d


def iterate_files(org_path: str, additional_path: str):
    folders = sorted(os.listdir(additional_path))
    folders_org = sorted(os.listdir(org_path))

    # check if all folders are present
    assert folders == folders_org, 'Folders do not match'

    for folder in folders:
        # get names of all yaml files in the additional path folder
        yaml_files = sorted([x for x in os.listdir(os.path.join(additional_path, folder)) if x.endswith('.yaml')])
        # get timestamps for yaml files {timestamp}_all_agents.yaml
        timestamps = [x.split('_')[0] for x in yaml_files]

        # get all folders in the original path (cav_id)
        cav_folders = sorted([x for x in os.listdir(os.path.join(org_path, folder)) if x.isdigit()])

        # iterate timestamps
        for timestamp in timestamps:
            # load additional all_agents yaml
            yaml_file = os.path.join(additional_path, folder, f'{timestamp}_all_agents.yaml')
            yaml_content = yaml_utils.load_yaml(yaml_file)
            # get vehicles
            vehicles = yaml_content['vehicles']
            # iterate cav folders
            for cav_id in cav_folders:
                cav_path = os.path.join(org_path, folder, cav_id)
                # load the pcd file
                pcd_file = os.path.join(cav_path, f'{timestamp}.pcd')
                pcd = load_pcd_file(pcd_file)
                # load cav yaml
                cav_yaml_file = os.path.join(cav_path, f'{timestamp}.yaml')
                lidar_pose = yaml_utils.load_yaml(cav_yaml_file)['lidar_pose']

                # calculate lidar hits per vehicle
                calculate_lidar_hits(pcd, lidar_pose, vehicles)
                




if __name__ == '__main__':
    original_path = r'/data/public_datasets/OPV2V/original/train'
    additional_path = r'/data/public_datasets/OPV2V/original/train_test/additional'

    iterate_files(original_path, additional_path)
