import numpy as np
import cv2
import os
from kitti_helpers import calculate_occlusion_stats, calculate_occlusion, calc_bbox_height
from kitti_helpers import calc_projected_2d_bbox, crop_boxes_in_canvas, calculate_truncation
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


def add_kitti_metrics(vehicles_data, image, depth_map):
    window_width = image.shape[1]
    window_height = image.shape[0]
    for vehicle in vehicles_data:
        if vehicles_data[vehicle]['bbox'] is None or vehicles_data[vehicle]['num_visible_vertices'] < 1:
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
            additional_yaml_content = yaml_utils.load_yaml(yaml_file, use_cloader=True)
            # get vehicles
            all_vehicles = additional_yaml_content['vehicles']
            # iterate cav folders
            for cav_id in cav_folders:
                if folder == '2021_08_21_09_28_12' and timestamp == '000295':
                    print('debug')
                else:
                    continue
                cav_path = os.path.join(org_path, folder, cav_id)
                additional_cav_path = os.path.join(additional_path, folder, cav_id)
                # load cav yaml
                cav_yaml_file = os.path.join(cav_path, f'{timestamp}.yaml')
                cav_yaml_content = yaml_utils.load_yaml(cav_yaml_file, use_cloader=True)
                # load additional cav yaml
                additional_cav_yaml_file = os.path.join(additional_cav_path, f'{timestamp}.yaml')
                additional_cav_yaml_content = yaml_utils.load_yaml(additional_cav_yaml_file, use_cloader=True)
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
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['bbox_height'] = camera_visibility_vehicles_info[vehicle_id]['bbox_height']
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['occlusion'] = camera_visibility_vehicles_info[vehicle_id]['occlusion']
                        additional_cav_yaml_content['vehicles'][vehicle_id_int][rgb_name]['truncation'] = camera_visibility_vehicles_info[vehicle_id]['truncation']

                if folder == '2021_08_21_09_28_12' and timestamp == '000295':
                    print('debug')
                # update the additional cav yaml with camera info
                # save_updated_yaml(additional_cav_yaml_content, additional_cav_yaml_file)


def save_updated_yaml(cav_yaml_content, new_yaml_file_path):
    yaml_utils.save_yaml(cav_yaml_content, new_yaml_file_path)


if __name__ == "__main__":
    # original_path = r'/data/public_datasets/OPV2V/original/train'
    # additional_path = r'/data/public_datasets/OPV2V/original/train/additional'

    # update_yaml_with_camera_metrics(original_path, additional_path)

    # original_path = r'/data/public_datasets/OPV2V/original/validate'
    # additional_path = r'/data/public_datasets/OPV2V/original/validate/additional'

    # update_yaml_with_camera_metrics(original_path, additional_path)

    original_path = r'/data/public_datasets/OPV2V/original/test'
    additional_path = r'/data/public_datasets/OPV2V/original/test/additional'

    update_yaml_with_camera_metrics(original_path, additional_path)
