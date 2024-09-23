import numpy as np
import yaml
import cv2
import os
import matplotlib.pyplot as plt
from kitti_helpers import calculate_occlusion_stats, calculate_occlusion, calc_bbox_height
from kitti_helpers import point_in_canvas, calc_projected_2d_bbox, crop_boxes_in_canvas, calculate_truncation
import tqdm
import opencood.hypes_yaml.yaml_utils as yaml_utils
from project_boxes_to_image import x_to_world, create_3d_box, get_image_points, actor_in_front_of_camera


def read_depth_image(path: str, style='original') -> np.ndarray:
    """
    Read depth image generated from the CARLA simulator and return the depth values as a numpy array.

    :param path: Path to the depth image.
    :param style: The style of depth values to return ('original' or 'normalized').
    :return: Depth values as a numpy array.
    """

    # Read the depth image as is (unchanged)
    raw_depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if raw_depth_image is None:
        raise FileNotFoundError(f"Could not read the depth image from path: {path}")
    
    # CARLA uses 24-bit depth encoded in RGB channels: R, G, B
    # Decode depth values using CARLA's encoding formula: depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    raw_depth_image = raw_depth_image.astype(np.float32)
    depth_image = (raw_depth_image[:, :, 2] + 
                   raw_depth_image[:, :, 1] * 256.0 + 
                   raw_depth_image[:, :, 0] * 256.0 * 256.0) / (256.0 ** 3 - 1)
    depth_image *= 1000  # Scale to 1000 meters as specified by CARLA's max render distance

    if style == 'original':
        # # Plot the image with matplotlib
        # clipped_depth_image = np.clip(depth_image, 0, 100)  # Clip the depth values to 500 meters
        # plt.imshow(clipped_depth_image, cmap='gray')
        # plt.colorbar()
        # plt.savefig('current_depth_image.png')
        # plt.close()
        # # Return the raw depth values in meters
        return depth_image

    elif style == 'normalized':
        # Normalize the depth values to a 0-1 range
        depth_image_normalized = cv2.normalize(depth_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return depth_image_normalized

    else:
        raise ValueError(f"Style '{style}' not implemented")


def read_rgb_image(path: str) -> np.ndarray:
    """
    Read RGB image generated from the simulator and return the image as a numpy array

    :param path: Path to the RGB image
    """

    rgb_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return rgb_image

def get_transformation_matrix(location, rotation):
    """
    Creates a transformation matrix from location and rotation (roll, yaw, pitch).
    """
    # Assuming the order of rotation is roll, yaw, pitch (rotation[0] = roll, rotation[1] = yaw, rotation[2] = pitch)
    c_r = np.cos(np.radians(rotation[0]))  # Roll
    s_r = np.sin(np.radians(rotation[0]))
    c_y = np.cos(np.radians(rotation[1]))  # Yaw
    s_y = np.sin(np.radians(rotation[1]))
    c_p = np.cos(np.radians(rotation[2]))  # Pitch
    s_p = np.sin(np.radians(rotation[2]))

    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location[0] # x
    matrix[1, 3] = location[1] # y
    matrix[2, 3] = location[2] # z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def create_bounding_box_points(extent):
    """
    Returns the 3D coordinates of the vehicle's bounding box in its local coordinate system.
    """
    cords = np.zeros((8, 4))
    cords[0, :] = np.array([extent[0], extent[1], 0, 1])
    cords[1, :] = np.array([-extent[0], extent[1], 0, 1])
    cords[2, :] = np.array([-extent[0], -extent[1], 0, 1])
    cords[3, :] = np.array([extent[0], -extent[1], 0, 1])
    cords[4, :] = np.array([extent[0], extent[1], 2*extent[2], 1])
    cords[5, :] = np.array([-extent[0], extent[1], 2*extent[2], 1])
    cords[6, :] = np.array([-extent[0], -extent[1], 2*extent[2], 1])
    cords[7, :] = np.array([extent[0], -extent[1], 2*extent[2], 1])
    return cords

def get_bounding_box(vehicle_info, camera_info):
    # Vehicle Transformation
    location = vehicle_info['location'] # x, y, z
    rotation = vehicle_info['angle'] # roll, yaw, pitch
    vehicle_matrix = get_transformation_matrix(location, rotation)

    # Camera Transformation
    location = camera_info['cords'][:3] # x, y, z
    rotation = camera_info['cords'][3:] # roll, yaw, pitch
    camera_matrix = get_transformation_matrix(location, rotation)
    world_to_camera = np.linalg.inv(camera_matrix)
    
    # Create the bounding box in vehicle coordinates
    extent = vehicle_info['extent']
    bounding_box = create_bounding_box_points(extent)

    # Transform bounding box to from vehicle to world coordinates
    bb_world_cords = np.dot(vehicle_matrix, bounding_box.T)

    # Transform bounding box from world to camera coordinates
    bb_camera_cords = np.dot(world_to_camera, bb_world_cords).T
    camera_bbox = bb_camera_cords[:, :3]

    # Project to camera plane (x, y ,z) -> (y, -z, x)
    camera_bbox = np.array([camera_bbox[:, 1], -camera_bbox[:, 2], camera_bbox[:, 0]]).T
    camera_bbox = camera_bbox.squeeze()
    # Project the bounding box to the image plane
    points2d = np.dot(camera_info['intrinsic'], camera_bbox.T)

    points_2d_depth = points2d[2, :]
    points_2d = (points2d[:2, :] / points_2d_depth).T

    # Stack the 2D points with their depth values to form [u, v, depth]
    points_2d_depth = points_2d_depth.reshape(-1, 1)
    points_2d_with_depth = np.hstack((points_2d, points_2d_depth))

    return points_2d_with_depth

def get_points_in_canvas(vehicles, rgb_image):
    points_list = []
    for vehicle in vehicles:
        for bbox_point in vehicles[vehicle]['bbox']:    
            x_2d = bbox_point[0]
            y_2d = bbox_point[1]
            point_depth = bbox_point[2]

            # Adjust window check and ensure (u, v) are within valid ranges
            if 100 > point_depth > 0 and point_in_canvas((x_2d, y_2d), window_height=rgb_image.shape[0], window_width=rgb_image.shape[1]):
                points_list.append([x_2d, y_2d])
    
    # Iterate over the points and add them as red dots to the image
    for point in points_list:
        cv2.circle(rgb_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    
    # Save the image with the points
    cv2.imwrite('points.png', rgb_image)
    

def post_process(data_path: str, overwrite: bool = False):
    """
    Post-process the data generated from the simulator

    :param data_path: Path to the data directory holding all datapoints
    :param overwrite: Overwrite the existing data or create new yaml file
    """
    results = {}
    for datapoint in os.listdir(data_path):
        results[datapoint] = {}
        # open the .yaml file
        data = yaml.load(open(os.path.join(data_path, datapoint, f'{datapoint}.yaml'), 'r'), Loader=yaml.FullLoader)
        # read the depth image
        depth_image = read_depth_image(os.path.join(data_path, datapoint, f'depth_raw.png'))
        # read the rgb image
        rgb_image = read_rgb_image(os.path.join(data_path, datapoint, f'rgb.png'))
        # create a vehicle dict from the data
        cameras = [key for key in data.keys() if 'camera' in key]
        for camera in cameras:
            # get the camera calibration
            camera_info = data[camera]
            vehicles_data = get_bbox(data['vehicles'], camera_info, depth_image)
            vehicles_data = calculate_occlusion_stats(rgb_image, depth_image, vehicles_data)
            vehicles_data = add_kitti_metrics(vehicles_data, rgb_image, depth_image)
            results[datapoint][camera] = vehicles_data
    return results

def add_kitti_metrics(vehicles_data, image, depth_map):
    window_width = image.shape[1]
    window_height = image.shape[0]
    for vehicle in vehicles_data:
        if vehicles_data[vehicle]['bbox'] is None or vehicles_data[vehicle]['num_visible_vertices'] <1:
            vehicles_data[vehicle]['bbox_height'] = None
            vehicles_data[vehicle]['occlusion'] = None
            vehicles_data[vehicle]['truncation'] = None
        else:
            camera_bbox = vehicles_data[vehicle]['bbox']
            uncropped_bbox_2d = calc_projected_2d_bbox(camera_bbox)
            camera_bbox = crop_boxes_in_canvas(camera_bbox, window_width, window_height)
            bbox_2d = calc_projected_2d_bbox(camera_bbox)
            height = calc_bbox_height(bbox_2d)
            occlusion, raw_occlusion = calculate_occlusion(camera_bbox, vehicles_data[vehicle]['extent'], depth_map, window_width, window_height)
            truncation = calculate_truncation(uncropped_bbox_2d, bbox_2d)

            # write the results to the vehicle dict
            vehicles_data[vehicle]['bbox_height'] = height
            vehicles_data[vehicle]['occlusion'] = raw_occlusion
            vehicles_data[vehicle]['truncation'] = truncation
    return vehicles_data


def get_bbox(vehicles, camera_info, image):
    vehicle_bbox = {}
    camera0 = camera_info
    # get camera intrinsic
    camera_pose = camera0['cords']
    camera_extrinsic = x_to_world(camera_pose)
    world_to_camera = np.linalg.inv(camera_extrinsic)

    intrinsic = camera0['intrinsic']

    for vehicle_id in vehicles:
        vehicle_bbox[vehicle_id] = {}
        vehicle = vehicles[vehicle_id]
        v_angle = np.asarray(vehicle['angle'])
        v_location = np.asarray(vehicle['location'])
        extent = np.asarray(vehicle['extent'])
        vehicle_bbox[vehicle_id]['extent'] = extent

        # check if vehicle is in front of camera
        if not actor_in_front_of_camera(camera_pose, v_location):
            vehicle_bbox[vehicle_id]['bbox'] = None
            continue

        box_3d = create_3d_box(extent, v_angle)
        box_3d += v_location

        # get image point for each box point
        image_points = get_image_points(intrinsic, world_to_camera, box_3d)
        vehicle_bbox[vehicle_id]['bbox'] = image_points.T

        # draw box
        # image = draw_box(image, image_points)

    # save image
    # cv2.imwrite('bbox_image.png', image)

    return vehicle_bbox


def load_rgb_file(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_depth_file(path):
    raw_depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # CARLA uses 24-bit depth encoded in RGB channels: R, G, B
    # Decode depth values using CARLA's encoding formula: depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    raw_depth_image = raw_depth_image.astype(np.float32)
    depth_image = (raw_depth_image[:, :, 2] + 
                   raw_depth_image[:, :, 1] * 256.0 + 
                   raw_depth_image[:, :, 0] * 256.0 * 256.0) / (256.0 ** 3 - 1)
    depth_image *= 1000  # Scale to 1000 meters as specified by CARLA's max render distance

    return depth_image


def calculate_camera_visibility(rgb_image, depth_image, rgb_camera_info, vehicles):
    vehicles_data = get_bbox(vehicles, rgb_camera_info, depth_image)
    vehicles_data = calculate_occlusion_stats(rgb_image, depth_image, vehicles_data)
    vehicles_data = add_kitti_metrics(vehicles_data, rgb_image, depth_image)
    
    return vehicles_data


def update_yaml_with_camera_metrics(org_path: str, additional_path: str):
    folders = sorted(os.listdir(additional_path))
    folders_org = sorted(os.listdir(org_path))
    # delete 'additional' folder from folders_org
    folders_org.remove('additional')
    # delete all none folders from 'folders' and 'folders_org'
    folders = [x for x in folders if os.path.isdir(os.path.join(additional_path, x))]
    folders_org = [x for x in folders_org if os.path.isdir(os.path.join(org_path, x))]

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
                additional_cav_path = os.path.join(additional_path, folder, cav_id)
                # load cav yaml
                cav_yaml_file = os.path.join(cav_path, f'{timestamp}.yaml')
                cav_yaml_content = yaml_utils.load_yaml(cav_yaml_file)
                # load additional cav yaml
                additional_cav_yaml_file = os.path.join(additional_cav_path, f'{timestamp}.yaml')
                additional_cav_yaml_content = yaml_utils.load_yaml(additional_cav_yaml_file)
                # Load the camera files
                for rgb_name, depth_name in [['camera0', 'front'], ['camera1', 'right'], ['camera2', 'left'], ['camera3', 'back']]:
                    camera_info = cav_yaml_content[rgb_name]
                    # load rgb image
                    rgb_image = load_rgb_file(os.path.join(cav_path, f'{timestamp}_{rgb_name}.png'))
                    # load depth image
                    depth_image = load_depth_file(os.path.join(additional_cav_path, f'{timestamp}_depth_camera_{depth_name}.png'))

                    camera_visibility_vehicles_info = calculate_camera_visibility(
                        rgb_image, depth_image, camera_info, all_vehicles
                    )

                    # update the yaml with the new metrics
                    for vehicle_id in camera_visibility_vehicles_info:
                        vehicle_id_int = int(vehicle_id)
                        cav_id_int = int(cav_id)
                        # skip cav_id
                        if vehicle_id_int == cav_id_int:
                            continue
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name] = {}
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['camera_bbox_height'] = camera_visibility_vehicles_info[vehicle_id]['bbox_height']
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['camera_occlusion'] = camera_visibility_vehicles_info[vehicle_id]['occlusion']
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['camera_truncation'] = camera_visibility_vehicles_info[vehicle_id]['truncation']

            # update the additional cav yaml with camera info
            save_updated_yaml(additional_cav_yaml_content, additional_cav_yaml_file)

def save_updated_yaml(cav_yaml_content, new_yaml_file_path):
    yaml_utils.save_yaml(cav_yaml_content, new_yaml_file_path)


if __name__ == "__main__":
    original_path = r'/data/public_datasets/OPV2V/original/train'
    additional_path = r'/data/public_datasets/OPV2V/original/train/additional'

    update_yaml_with_camera_metrics(original_path, additional_path)

    original_path = r'/data/public_datasets/OPV2V/original/validate'
    additional_path = r'/data/public_datasets/OPV2V/original/validate/additional'

    update_yaml_with_camera_metrics(original_path, additional_path)

    original_path = r'/data/public_datasets/OPV2V/original/test'
    additional_path = r'/data/public_datasets/OPV2V/original/test/additional'

    update_yaml_with_camera_metrics(original_path, additional_path)
