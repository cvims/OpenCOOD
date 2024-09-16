import numpy as np
from typing import Tuple
import numpy as np
from itertools import product


def point_in_canvas(pos, window_height, window_width):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < window_height) and (pos[1] >= 0) and (pos[1] < window_width):
        return True
    return False

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
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy+y, dx+x), window_height=window_height-5, window_width=window_width-5):
            # If the depth map says the pixel is closer to the camera than the actual vertex
            if depth_map[y+dy, x+dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return any(is_occluded)

def calculate_occlusion_stats(image, depth_map, vehicle_dict, max_render_depth=1000):
    """ Draws each vertex in vertices_pos2d if it is in front of the camera
        The color is based on whether the object is occluded or not.
        Returns the number of visible vertices and the number of vertices outside the camera.
    """
    vehicle_keys = vehicle_dict.keys()
    for vehicle in vehicle_keys:
        num_visible_vertices = 0
        num_vertices_outside_camera = 0
        if vehicle_dict[vehicle]['bbox'] is None:
            vehicle_dict[vehicle]['num_visible_vertices'] = 0
        else:
            bbox_points = vehicle_dict[vehicle]['bbox']
            for i in range(len(bbox_points)):
                x_2d = int(bbox_points[i, 0])
                y_2d = int(bbox_points[i, 1])
                point_depth = bbox_points[i, 2]

                # if the point is in front of the camera but not too far away
                if max_render_depth > point_depth > 0 and point_in_canvas((y_2d, x_2d),window_height=image.shape[0], window_width=image.shape[1]):
                    is_occluded = point_is_occluded(
                        (y_2d, x_2d), point_depth, depth_map, window_height=image.shape[0], window_width=image.shape[1])
                    if is_occluded:
                        pass
                        # vertex_color = OCCLUDED_VERTEX_COLOR
                    else:
                        num_visible_vertices += 1
                        # vertex_color = VISIBLE_VERTEX_COLOR
                    # if draw_vertices:
                    #     draw_rect(image, (y_2d, x_2d), 4, vertex_color)
                else:
                    num_vertices_outside_camera += 1
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
    neg_x_inds = np.where(cam_bboxes[:, 0] < 0)[0]
    out_x_inds = np.where(cam_bboxes[:, 0] > window_width)[0]
    neg_y_inds = np.where(cam_bboxes[:, 1] < 0)[0]
    out_y_inds = np.where(cam_bboxes[:, 1] > window_height)[0]
    cam_bboxes[neg_x_inds, 0] = 0
    cam_bboxes[out_x_inds, 0] = window_width
    cam_bboxes[neg_y_inds, 1] = 0
    cam_bboxes[out_y_inds, 1] = window_height
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

    #depth_margin should depend on the rotation of the object but this solution works fine
    depth_margin = np.max([2*width, 2*length])
    is_occluded = []

    for x in range(int(min_x), int(max_x)):
        for y in range(int(min_y), int(max_y)):
            is_occluded.append(point_is_occluded(
                (y, x), bbox_3d_mid - depth_margin, depth_map, window_height, window_width))

    raw_occlusion = ((float(np.sum(is_occluded))) / ((max_x-min_x) * (max_y-min_y)))

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