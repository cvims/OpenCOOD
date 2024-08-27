import numpy as np


def get_camera_intrinsic(sensor, to_list: bool = True):
    """
    Retrieve the camera intrinsic matrix.

    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.

    Returns
    -------
    matrix_x : np.ndarray
        The 2d intrinsic matrix.
        If to_list is True, return a list of intrinsic matrix.

    """
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    if to_list:
        return matrix_k.tolist()

    return matrix_k
