import os
import opencood.hypes_yaml.yaml_utils as yaml_utils
import cv2
import numpy as np


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
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


def create_3d_box(extent, angle):
    """
    Create 3D box from extent and angle

    Parameters
    ----------
    extent : list
        [length, width, height]
    angle : float
        The angle of the box

    Returns
    -------
    box_3d : np.ndarray
        The 3D box
    """
    length, width, height = extent
    box_3d = np.array([[length, -width, 0],
                        [length, width, 0],
                        [-length, width, 0],
                        [-length, -width, 0],
                        [length, -width, height * 2],
                        [length, width, height * 2],
                        [-length, width, height * 2],
                        [-length, -width, height * 2]])

    # rotate box
    yaw_rad = np.radians(angle[1])
    rotation_matrix = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                [0, 0, 1]])

    box_3d = np.dot(box_3d, rotation_matrix)

    return box_3d


def get_image_points(K, E, box_3d):
    """
    Get image points from 3D box

    Parameters
    ----------
    K : np.ndarray
        The intrinsic matrix
    E : np.ndarray
        The extrinsic matrix
    box_3d : np.ndarray
        The 3D box

    Returns
    -------
    image_points : np.ndarray
        The image points
    """
    # project 3D box to camera coordinate
    box_3d = np.vstack([box_3d.T, np.ones(8)])  # 4x8
    box_3d = np.dot(E, box_3d)

    # coordinate system swap from UE4 to OpenCV
    box_3d = np.array([box_3d[1, :], -box_3d[2, :], box_3d[0, :]])

    # project 3D box to image plane
    image_points = np.dot(K, box_3d)

    # normalize
    image_points /= image_points[2, :]

    return image_points


def draw_box(image, points):
    """
    Draw box on image

    Parameters
    ----------
    image : np.ndarray
        The image
    points : np.ndarray
        The image points

    Returns
    -------
    image : np.ndarray
        The image with box
    """
    # draw lines
    for i in range(4):
        i_next = (i + 1) % 4
        image = cv2.line(image, (int(points[0, i]), int(points[1, i])),
                         (int(points[0, i_next]), int(points[1, i_next])), (0, 255, 0), 2)
        image = cv2.line(image, (int(points[0, i + 4]), int(points[1, i + 4])),
                         (int(points[0, i_next + 4]), int(points[1, i_next + 4])), (0, 255, 0), 2)
        image = cv2.line(image, (int(points[0, i]), int(points[1, i])),
                         (int(points[0, i + 4]), int(points[1, i + 4])), (0, 255, 0), 2)

    return image


def actor_in_front_of_camera(camera_pose, actor_location):
    camera_yaw_rad = np.radians(camera_pose[4])
    forward_vector = np.array([np.cos(camera_yaw_rad), np.sin(camera_yaw_rad)])
    actor_vector = np.array(actor_location[:2]) - np.array(camera_pose[:2])
    dot_product = np.dot(forward_vector, actor_vector)

    return dot_product > 0


if __name__ == '__main__':
    yaml_path = r'/home/dominik/Git_Repos/Private/OpenCOOD/test/000069.yaml'
    rgb_path = r'/home/dominik/Git_Repos/Private/OpenCOOD/test/rgb.png'

    # load yaml
    yaml_content = yaml_utils.load_yaml(yaml_path)

    # load image
    image = cv2.imread(rgb_path)

    # get camera0 attributes
    camera0 = yaml_content['camera0']
    # vehicle position
    vehicle_pose = yaml_content['true_ego_pos']
    # visible vehicles
    vehicles = yaml_content['vehicles']

    # get camera intrinsic
    camera_pose = camera0['cords']
    camera_extrinsic = x_to_world(camera_pose)
    world_to_camera = np.linalg.inv(camera_extrinsic)

    intrinsic = camera0['intrinsic']

    for vehicle_id in vehicles:
        vehicle = vehicles[vehicle_id]
        v_angle = np.asarray(vehicle['angle'])
        v_location = np.asarray(vehicle['location'])
        extent = np.asarray(vehicle['extent'])

        # check if vehicle is in front of camera
        if not actor_in_front_of_camera(camera_pose, v_location):
            continue

        box_3d = create_3d_box(extent, v_angle)
        box_3d += v_location

        # get image point for each box point
        image_points = get_image_points(intrinsic, world_to_camera, box_3d)

        # draw box
        image = draw_box(image, image_points)


    # save image
    cv2.imwrite('output.png', image)
