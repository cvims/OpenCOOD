import opencood.hypes_yaml.yaml_utils as yaml_utils
import open3d as o3d
import os
import numpy as np
import copy
from numba import njit, types, typed
import tqdm


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
    # clone pcd
    combined_pcd = copy.deepcopy(pcd)

    for v_3d_box in v_3d_boxes:
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

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


@njit
def calculate_hits(pcd_points, bbox_vals):
    # Initialize numba typed.Dict for v_hits
    v_hits = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )

    for point in np.asarray(pcd_points):
    # Iterate through the 3D boxes and check for a hit
        for v_id in bbox_vals.keys():
            bbox_val = bbox_vals[v_id]
            if point_in_3d_bounding_box(point, bbox_val[0], bbox_val[1], bbox_val[2], bbox_val[3], bbox_val[4], bbox_val[5]):
                if v_id in v_hits:
                    v_hits[v_id] += 1
                else:
                    v_hits[v_id] = 1
                break
    
    return v_hits


@njit
def point_in_3d_bounding_box(point, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z):
    """
    Check if a point is inside a 3D bounding box
    :param point: point
    :param bounding_box_3d: 3D bounding box coordinates
        format: np.array([[x1, y1, z1], [x2, y2, z2], ...])
    """
    offset = 0.01

    if bbox_min_x - offset <= point[0] <= bbox_max_x + offset and \
        bbox_min_y - offset <= point[1] <= bbox_max_y + offset and \
        bbox_min_z - offset <= point[2] <= bbox_max_z + offset:
        return True

    return False


def calculate_lidar_hits(pcd, lidar_pose, vehicles):
    # x,y,z
    lidar_position = np.asarray(lidar_pose[:3]) # world coordinates (x,y,z)
    lidar_rotation = np.asarray(lidar_pose[3:]) # pitch, yaw, roll

    # calculate the world coordinates of the pcd points (consider rotation and translation)
    # pcd_points = np.asarray(pcd.points)
    pcd_points = pcd
    lidar_yaw = np.deg2rad(lidar_rotation[1])
    R = np.array([
        [np.cos(lidar_yaw), -np.sin(lidar_yaw), 0],
        [np.sin(lidar_yaw), np.cos(lidar_yaw), 0],
        [0, 0, 1]
    ])

    pcd_points = np.dot(R, pcd_points.T).T
    pcd_points += lidar_position

    # to o3d pcd
    # _pcd = o3d.geometry.PointCloud()
    # _pcd.points = o3d.utility.Vector3dVector(pcd_points)

    v_3d_boxes = {}
    for v_id, v_data in vehicles.items():
        angle = v_data['angle'] # pitch, yaw, roll
        extent = v_data['extent']
        location = v_data['location'] # world coordinates

        v_3d_bbox = generate_3d_bbox(extent, location, np.deg2rad(angle[1]))
        # add lidar position to the 3d bounding box
        # v_3d_bbox -= lidar_position
        v_3d_boxes[int(v_id)] = v_3d_bbox
    
    # draw the 3d bounding boxes into point cloud
    #d_pcd = draw_3d_bbox(_pcd, v_3d_boxes.values())

    # save pcd
    #o3d.io.write_point_cloud('output.pcd', d_pcd)

    # precalculate the min and max values for the 3d bounding boxes (for faster access)
    bbox_vals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:]
    )

    for v_id, v_3d_box in v_3d_boxes.items():
        bbox_vals[v_id] = np.array([
            np.min(v_3d_box[:, 0]),
            np.min(v_3d_box[:, 1]),
            np.min(v_3d_box[:, 2]),
            np.max(v_3d_box[:, 0]),
            np.max(v_3d_box[:, 1]),
            np.max(v_3d_box[:, 2])
        ], dtype=np.float64)

    # measure time
    v_hits = calculate_hits(pcd_points, bbox_vals)

    # numba v_hits to regular dict
    v_hits = dict(v_hits)

    return v_hits


def generate_3d_bbox(extent, location, yaw_rad):
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

    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    bbox_3d = np.dot(R, bbox_3d.T).T
    bbox_3d = bbox_3d + location

    return bbox_3d


def save_updated_yaml(cav_yaml_content, additional_yaml_content, vehicle_hits, cav_id, new_yaml_file_path):
    # we merge the vehicle section of {timestamp}.yaml with the {timestamp}_all_agents.yaml
    new_yaml_content = copy.deepcopy(cav_yaml_content)
    vehicle_yaml_content = new_yaml_content['vehicles']
    all_vehicle_yaml_content = additional_yaml_content['vehicles']

    # all_vehicle_yaml_content keys to int
    all_vehicle_yaml_content = {int(k): v for k, v in all_vehicle_yaml_content.items()}

    # delete the vehicles from all_vehicle_yaml_content that are in vehicle_yaml_content
    for v_id in vehicle_yaml_content.keys():
        if v_id in all_vehicle_yaml_content:
            del all_vehicle_yaml_content[v_id]
    
    # merge the two dictionaries
    vehicle_yaml_content.update(all_vehicle_yaml_content)

    # delete cav_id if it exists
    if int(cav_id) in vehicle_yaml_content:
        del vehicle_yaml_content[int(cav_id)]

    # add lidar hits to the yaml
    for v_id in vehicle_yaml_content.keys():
        if v_id in vehicle_hits:
            vehicle_yaml_content[v_id]['lidar_hits'] = vehicle_hits[v_id]
        else:
            vehicle_yaml_content[v_id]['lidar_hits'] = 0
    
    # update vehicle section in new_yaml_content
    new_yaml_content['vehicles'] = vehicle_yaml_content               

    # save the new yaml                
    yaml_utils.save_yaml(new_yaml_content, new_yaml_file_path)


def update_yaml_with_lidar_hits(org_path: str, additional_path: str):
    folders = sorted(os.listdir(additional_path))
    folders_org = sorted(os.listdir(org_path))
    # delete 'additional' folder from folders_org
    folders_org.remove('additional')

    # check if all folders are present
    assert folders == folders_org, 'Folders do not match'

    for folder in tqdm.tqdm(folders):
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
            additional_yaml_content = yaml_utils.load_yaml(yaml_file)
            # get vehicles
            all_vehicles = additional_yaml_content['vehicles']
            # iterate cav folders
            for cav_id in cav_folders:
                cav_path = os.path.join(org_path, folder, cav_id)
                # load the pcd file
                pcd_file = os.path.join(cav_path, f'{timestamp}.pcd')
                pcd = load_pcd_file(pcd_file)
                # load cav yaml
                cav_yaml_file = os.path.join(cav_path, f'{timestamp}.yaml')
                cav_yaml_content = yaml_utils.load_yaml(cav_yaml_file)
                lidar_pose = cav_yaml_content['lidar_pose']

                # calculate lidar hits per vehicle
                vehicle_hits = calculate_lidar_hits(np.asarray(pcd.points), lidar_pose, all_vehicles)

                # save new yaml file
                new_yaml_file_path = os.path.join(additional_path, folder, str(cav_id), f'{timestamp}.yaml')
                save_updated_yaml(cav_yaml_content, additional_yaml_content, vehicle_hits, cav_id, new_yaml_file_path)
    
            

if __name__ == '__main__':
    original_path = r'/data/public_datasets/OPV2V/original/train'
    additional_path = r'/data/public_datasets/OPV2V/original/train/additional'

    update_yaml_with_lidar_hits(original_path, additional_path)
