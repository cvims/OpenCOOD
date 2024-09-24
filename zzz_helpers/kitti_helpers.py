import numpy as np
from typing import Tuple
import numpy as np
from itertools import product


def point_in_canvas(pos, window_height, window_width):
    """Return true if point is in canvas"""
    # if (pos[0] >= 0) and (pos[0] < window_height) and (pos[1] >= 0) and (pos[1] < window_width):
    #     return True
    # return False

    return 0 <= pos[0] < window_height and 0 <= pos[1] < window_width

def calc_projected_2d_bbox(vehicle_dict):
    """ Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
        Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
        Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    vehcile_keys = vehicle_dict.keys()
    for vehicle in vehcile_keys:
        vertices_pos2d = vehicle_dict[vehicle]['bbox']
        if vertices_pos2d is None:
            vehicle_dict[vehicle]['bbox'] = None
        else:
            x_coords = vertices_pos2d[:, 0]
            y_coords = vertices_pos2d[:, 1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            vehicle_dict[vehicle]['projected_bbox'] = [min_x, min_y, max_x, max_y]

def point_is_occluded(point: Tuple[float], vertex_depth: float, depth_map: np.ndarray, window_height: int, window_width: int):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    y, x = map(int, point)

    # Define neighbor shifts: top-left, top-right, bottom-left, bottom-right
    neighbors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    # Calculate the neighbor positions
    neighbor_positions = neighbors + np.array([y, x])

    # Clip the neighbors' positions to stay within the canvas bounds
    neighbor_positions[:, 0] = np.clip(neighbor_positions[:, 0], 0, window_height - 5)
    neighbor_positions[:, 1] = np.clip(neighbor_positions[:, 1], 0, window_width - 5)

    # Extract the depth values of the neighbors
    neighbor_depths = depth_map[neighbor_positions[:, 0], neighbor_positions[:, 1]]

    # Check if any of the neighbors have a depth less than the vertex depth
    return np.any(neighbor_depths < vertex_depth)

def calculate_occlusion_stats(image, depth_map, vehicle_dict, max_render_depth=1000):
    """ Draws each vertex in vertices_pos2d if it is in front of the camera
        The color is based on whether the object is occluded or not.
        Returns the number of visible vertices and the number of vertices outside the camera.
    """
    window_height, window_width = image.shape[0], image.shape[1]
    for vehicle, vehicle_data in vehicle_dict.items():
        if vehicle_dict[vehicle]['bbox'] is None:
            vehicle_dict[vehicle]['num_visible_vertices'] = 0
            continue

        bbox_points = vehicle_data['bbox']
        num_vertices = bbox_points.shape[0]

        # Initialize visible and out-of-camera vertex counters
        num_visible_vertices = 0
        num_vertices_outside_camera = 0

        # Extract 2D coordinates and depth values from bbox
        x_2d = bbox_points[:, 0].astype(int)
        y_2d = bbox_points[:, 1].astype(int)
        point_depths = bbox_points[:, 2]

        # Check if the points are within renderable depth and within the camera view
        valid_depths = (point_depths > 0) & (point_depths < max_render_depth)
        in_canvas = (0 <= x_2d) & (x_2d < window_width) & (0 <= y_2d) & (y_2d < window_height)

        # Points that are valid and in the camera view
        valid_points = valid_depths & in_canvas

        # Iterate over the valid points
        for idx in np.where(valid_points)[0]:
            is_occluded = point_is_occluded(
                (y_2d[idx], x_2d[idx]), point_depths[idx], depth_map, window_height, window_width
            )
            if not is_occluded:
                num_visible_vertices += 1
            # Optionally draw the point based on visibility here
            # if draw_vertices:
            #     vertex_color = VISIBLE_VERTEX_COLOR if not is_occluded else OCCLUDED_VERTEX_COLOR
            #     draw_rect(image, (y_2d[idx], x_2d[idx]), 4, vertex_color)

        # The rest of the vertices are outside the camera
        num_vertices_outside_camera = num_vertices - np.sum(valid_points)

        # Update the vehicle dictionary
        vehicle_dict[vehicle]['num_visible_vertices'] = num_visible_vertices

    return vehicle_dict

def calc_projected_2d_bbox(vertices_pos2d):
    """ Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
        Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
        Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    x_coords = vertices_pos2d[:, 0]
    y_coords = vertices_pos2d[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return [min_x, min_y, max_x, max_y]

def crop_boxes_in_canvas(cam_bboxes, window_width, window_height):
    # neg_x_inds = np.where(cam_bboxes[:, 0] < 0)[0]
    # out_x_inds = np.where(cam_bboxes[:, 0] > window_width)[0]
    # neg_y_inds = np.where(cam_bboxes[:, 1] < 0)[0]
    # out_y_inds = np.where(cam_bboxes[:, 1] > window_height)[0]
    # cam_bboxes[neg_x_inds, 0] = 0
    # cam_bboxes[out_x_inds, 0] = window_width
    # cam_bboxes[neg_y_inds, 1] = 0
    # cam_bboxes[out_y_inds, 1] = window_height

    # Clamp the x and y coordinates between 0 and the window dimensions
    cam_bboxes[:, 0] = np.clip(cam_bboxes[:, 0], 0, window_width)  # Clamp x coordinates
    cam_bboxes[:, 1] = np.clip(cam_bboxes[:, 1], 0, window_height)  # Clamp y coordinates

    return cam_bboxes

def calc_bbox_height(bbox_2d):
    """ Calculate the height of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    return bbox_2d[3] - bbox_2d[1]

def calculate_occlusion(bbox, extent, depth_map, window_width, window_height):
    """Calculate the occlusion value of a 2D bounding box.
    Iterate through each point (pixel) in the bounding box and declare it occluded only
    if the 4 surroinding points (pixels) are closer to the camera (by using the help of depth map)
    than the actual distance to the middle of the 3D bounding boxe and some margin (the extent of the object)
    """
    bbox_3d_mid = np.mean(bbox[:,2])
    min_x, min_y, max_x, max_y = calc_projected_2d_bbox(bbox)

    height, width, length = extent

    # depth_margin should depend on the rotation of the object but this solution works fine
    depth_margin = np.max([2*width, 2*length])

    # Create arrays for the x and y coordinates
    x_range = np.arange(int(min_x), int(max_x))
    y_range = np.arange(int(min_y), int(max_y))
    
    # Create a meshgrid for x and y
    x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='ij')
    
    # Flatten the arrays for processing
    flat_x = x_grid.flatten()
    flat_y = y_grid.flatten()

    # Prepare for checking occlusion
    vertex_depth = bbox_3d_mid - depth_margin
    occlusion_mask = np.zeros_like(flat_x, dtype=bool)

    # Define neighbor shifts: top-left, top-right, bottom-left, bottom-right
    neighbors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    # Iterate through all points and check their neighbors
    for dx, dy in neighbors:
        neighbor_x = flat_x + dx
        neighbor_y = flat_y + dy
        
        # Clip to stay within canvas bounds
        valid_mask = (neighbor_x >= 0) & (neighbor_x < window_width - 5) & \
                     (neighbor_y >= 0) & (neighbor_y < window_height - 5)

        # Get depth values for valid neighbors
        neighbor_depths = np.full(flat_x.shape, np.inf)  # Default to a large value
        neighbor_depths[valid_mask] = depth_map[neighbor_y[valid_mask], neighbor_x[valid_mask]]

        # Update occlusion mask
        occlusion_mask |= (neighbor_depths < vertex_depth)

    # Calculate the area of the bounding box
    bbox_area = (max_x - min_x) * (max_y - min_y)

    # Check if the area is non-zero before performing the division
    if bbox_area > 0:
        raw_occlusion = float(np.sum(occlusion_mask)) / bbox_area
    else:
        raw_occlusion = 0

    #discretize the 0–1 occlusion value into KITTI’s {0,1,2,3,4} labels by equally dividing the interval into 4 parts
    occlusion = np.digitize(raw_occlusion, bins=[0.25, 0.50, 0.75, 0.9])

    return occlusion, raw_occlusion


def calculate_truncation(uncropped_bbox, cropped_bbox):
    "Calculate how much of the object’s 2D uncropped bounding box is outside the image boundary"

    area_cropped = calc_bbox2d_area(cropped_bbox)
    area_uncropped = calc_bbox2d_area(uncropped_bbox)
    truncation = 1.0 - float(area_cropped / area_uncropped)
    return truncation

def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)