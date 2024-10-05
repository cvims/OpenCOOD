KITTI_DETECTION_CATEGORY_ENUM = {
    'easy': 0,
    'moderate': 1,
    'hard': 2,
    'very_hard': 3,
    'none': 4
}


def default_none(value, default):
    return value if value is not None else default


# def set_category_by_lidar_hits(lidar_hits: int, category_config: dict):
#     easy = category_config['easy']['hits']
#     moderate = category_config['moderate']['hits']
#     hard = category_config['hard']['hits']
#     very_hard = category_config['very_hard']['hits']
#     none = category_config['none']['hits']

#     if lidar_hits >= easy:
#         return VISIBLITY_CATEGORY_ENUM['easy']
#     elif lidar_hits >= moderate:
#         return VISIBLITY_CATEGORY_ENUM['moderate']
#     elif lidar_hits >= hard:
#         return VISIBLITY_CATEGORY_ENUM['hard']
#     elif lidar_hits >= very_hard:
#         return VISIBLITY_CATEGORY_ENUM['very_hard']
#     else:
#         return VISIBLITY_CATEGORY_ENUM['none']


def set_category_by_camera_props(camera_props: dict, category_config: dict):
    cam_prop_occlusion = 0.0 if camera_props['occlusion'] is None else camera_props['occlusion']
    cam_prop_bbox_height = 0.0 if camera_props['bbox_height'] is None else camera_props['bbox_height']
    cam_prop_truncation = 0.0 if camera_props['truncation'] is None else camera_props['truncation']

    # List of categories in descending order of difficulty
    categories = list(category_config.keys())

    # Iterate over the categories and check conditions
    for category in categories:
        config = category_config[category]
        if (cam_prop_bbox_height >= config['bbox_height'] and 
            cam_prop_occlusion <= config['occlusion'] and 
            cam_prop_truncation <= config['truncation']):
            return KITTI_DETECTION_CATEGORY_ENUM[category]

    # If none of the categories match, return 'none'
    return KITTI_DETECTION_CATEGORY_ENUM['none']
    

# def categorize_vehicle_visibility_by_lidar_hits(vehicle_list: list, category_config: dict):
#     """
#     Categorize vehicles in the temporal vehicles list by the lidar hits.
#     """

#     for vehicle in vehicle_list.values():
#         vehicle['lidar_visibility'] = set_category_by_lidar_hits(vehicle['lidar_hits'], category_config)
    
#     return vehicle_list


# def categorize_vehicle_visibility_by_camera_props(vehicle_list: list, category_config: dict):
#     """
#     Categorize vehicles in the temporal vehicles list by the camera properties.
#     """
#     for vehicle in vehicle_list.values():
#         # the minimum (easiest) visibility is chosen
#         # e.g. one camera has a visibility of 'hard' and another camera has a visibility of 'easy'
#         # then the visibility of the vehicle is 'easy'
#         min_vehicle_visibility = VISIBLITY_CATEGORY_ENUM['none']

#         # each vehicle can be detected by multiple cameras
#         # filter by attribute camera{0,1,2,3}
#         for camera_vis_key in ['camera0', 'camera1', 'camera2', 'camera3']:  # static for opv2v
#             # check if vehicle has camera_vis_key
#             if camera_vis_key in vehicle:
#                 camera_visibility = vehicle[camera_vis_key]
#                 category_by_props = set_category_by_camera_props(camera_visibility, category_config)
#                 min_vehicle_visibility = min(min_vehicle_visibility, category_by_props)

#         vehicle['camera_visibility'] = min_vehicle_visibility
    
#     return vehicle_list


def categorize_by_kitti_criteria(vehicle_list: list, category_config: dict):
    """
    Categorize vehicles in the temporal vehicles list by the camera properties.
    """
    for vehicle in vehicle_list.values():
        # the minimum (easiest) visibility is chosen
        # e.g. one camera has a visibility of 'hard' and another camera has a visibility of 'easy'
        # then the visibility of the vehicle is 'easy'
        min_vehicle_detection_criteria = KITTI_DETECTION_CATEGORY_ENUM['none']

        # across all cameras
        max_height = 0
        min_occlusion = 1
        min_truncation = 1

        # each vehicle can be detected by multiple cameras
        # filter by attribute camera{0,1,2,3}
        for camera_vis_key in ['camera0', 'camera1', 'camera2', 'camera3']:  # static for opv2v
            # check if vehicle has camera_vis_key
            if camera_vis_key in vehicle:
                camera_visibility = vehicle[camera_vis_key]
                category_by_props = set_category_by_camera_props(camera_visibility, category_config)
                min_vehicle_detection_criteria = min(min_vehicle_detection_criteria, category_by_props)

                # get the maximum height, minimum occlusion and minimum truncation
                max_height = max(max_height, default_none(camera_visibility['bbox_height'], max_height))
                min_occlusion = min(min_occlusion, default_none(camera_visibility['occlusion'], min_occlusion))
                min_truncation = min(min_truncation, default_none(camera_visibility['truncation'], min_truncation))

        vehicle['kitti_criteria'] = min_vehicle_detection_criteria
        vehicle['kitti_criteria_props'] = {
            'bbox_height': max_height,
            'occlusion': min_occlusion,
            'truncation': min_truncation
        }
    
    return vehicle_list


def is_lower_or_equal_detection_criteria(vehicle_category: int, min_category: int):
    """
    Check if the vehicle category is lower or equal to the minimum category.
    """
    return vehicle_category <= min_category


# def filter_vehicles_by_category(vehicles: dict, min_category: int, is_camera: bool):
#     """
#     Filter vehicles by category.
#     :param vehicles: dict of vehicles
#     :param min_category: minimum category, e.g. 'hard' means that only vehicles with category 'hard' or lower will be returned
#     """
#     if is_camera:
#         category_key = 'camera_visibility'
#     else:
#         category_key = 'lidar_visibility'

#     filtered_entry = {}
#     for v_id, vehicle in vehicles.items():
#         if is_lower_or_equal_visibility_category(vehicle[category_key], min_category):
#             filtered_entry[v_id] = vehicle

#     return filtered_entry


def filter_vehicles_by_category(vehicles: dict, min_category: int):
    """
    Filter vehicles by category.
    :param vehicles: dict of vehicles
    :param min_category: minimum category, e.g. 'hard' means that only vehicles with category 'hard' or lower will be returned
    """
    category_key = 'kitti_criteria'
    filtered_entry = {}
    for v_id, vehicle in vehicles.items():
        if is_lower_or_equal_detection_criteria(vehicle[category_key], min_category):
            filtered_entry[v_id] = vehicle

    return filtered_entry


def filter_by_opv2v_original_visibility(vehicles: dict):
    """
    Filter vehicles by the original visibility category.
    """
    filtered_entry = {}
    for v_id, vehicle in vehicles.items():
        if vehicle['opv2v_visible']:
            filtered_entry[v_id] = vehicle

    return filtered_entry

# def update_temporal_vehicles_list(all_in_range_vehicles_list: list, temporal_vehicles_list: list):
#     """
#     All vehicles must be filtered before calling this function, e.g. with lidar or camera specific filters (see filter by lidar_hits or filter by camera_props).
#     We update the temporal vehicles list with the latest vehicles of each previos frame.
#     If a vehicle was visible in the previous frame, we simply add this one to the current frame.
#     An exception of adding the previous visible vehicles is when it is out of range from the ego vehicle.
#     """
#     # Initialize the updated list with the first frame's vehicle data
#     updated_temporal_vehicles_list = [temporal_vehicles_list[0]]

#     for vehicle in updated_temporal_vehicles_list[0].values():
#         vehicle['temporal_min_lidar_visibility'] = vehicle['lidar_visibility']
#         vehicle['temporal_min_camera_visibility'] = vehicle['camera_visibility']

#     for i in range(1, len(temporal_vehicles_list)):
#         current_frame = temporal_vehicles_list[i]
#         prev_frame = updated_temporal_vehicles_list[-1]

#         for vehicle in current_frame.values():
#             vehicle['temporal_min_lidar_visibility'] = vehicle['lidar_visibility']
#             vehicle['temporal_min_camera_visibility'] = vehicle['camera_visibility']

#         current_frame_ids = list(current_frame.keys())
#         prev_frame_ids = list(prev_frame.keys())

#         for v_id in prev_frame_ids:
#             if v_id not in current_frame_ids and v_id in all_in_range_vehicles_list[i]:
#                 # add the vehicle to the current frame if it was visible in the previous frame (with the coordinates of the current frame)
#                 current_frame[v_id] = all_in_range_vehicles_list[i][v_id]
#                 # Preserve the visibility data from the previous frame
#                 current_frame[v_id]['lidar_visibility'] = prev_frame[v_id].get('lidar_visibility')
#                 current_frame[v_id]['camera_visibility'] = prev_frame[v_id].get('camera_visibility')
        
#         # update visibility category. Always the lowest category in the temporal stream
#         for vehicle in current_frame.values():
#             # Update lidar visibility if applicable
#             vehicle['temporal_min_lidar_visibility'] = min(
#                 vehicle.get('temporal_min_lidar_visibility', vehicle['lidar_visibility']),
#                 vehicle['lidar_visibility']
#             )
#             # Update camera visibility if applicable
#             vehicle['temporal_min_camera_visibility'] = min(
#                 vehicle.get('temporal_min_camera_visibility', vehicle['camera_visibility']),
#                 vehicle['camera_visibility']
#             )
        
#         updated_temporal_vehicles_list.append(current_frame)
    
#     return updated_temporal_vehicles_list


def update_temporal_vehicles_list(all_in_range_vehicles_list: list, temporal_vehicles_list: list):
    """
    All vehicles must be filtered before calling this function, e.g. with lidar or camera specific filters (see filter by lidar_hits or filter by camera_props).
    We update the temporal vehicles list with the latest vehicles of each previos frame.
    If a vehicle was visible in the previous frame, we simply add this one to the current frame.
    An exception of adding the previous visible vehicles is when it is out of range from the ego vehicle.
    """
    # Initialize the updated list with the first frame's vehicle data
    updated_temporal_vehicles_list = [temporal_vehicles_list[0]]

    for vehicle in updated_temporal_vehicles_list[0].values():
        vehicle['temporal_kitti_criteria'] = vehicle['kitti_criteria']

    for i in range(1, len(temporal_vehicles_list)):
        current_frame = temporal_vehicles_list[i]
        prev_frame = updated_temporal_vehicles_list[-1]

        for vehicle in current_frame.values():
            vehicle['temporal_kitti_criteria'] = vehicle['kitti_criteria']

        current_frame_ids = list(current_frame.keys())
        prev_frame_ids = list(prev_frame.keys())

        for v_id in prev_frame_ids:
            if v_id not in current_frame_ids and v_id in all_in_range_vehicles_list[i]:
                # add the vehicle to the current frame if it was visible in the previous frame (with the coordinates of the current frame)
                current_frame[v_id] = all_in_range_vehicles_list[i][v_id]
                # Preserve the visibility data from the previous frame
                current_frame[v_id]['kitti_criteria'] = prev_frame[v_id].get('kitti_criteria')
        
        # update visibility category. Always the lowest category in the temporal stream
        for vehicle in current_frame.values():
            # Update lidar visibility if applicable
            vehicle['temporal_kitti_criteria'] = min(
                vehicle.get('temporal_kitti_criteria', vehicle['kitti_criteria']),
                vehicle['kitti_criteria']
            )
        
        updated_temporal_vehicles_list.append(current_frame)
    
    return updated_temporal_vehicles_list


def update_kitti_criteria(vehicle_cav1, vehicle_cav2, category_config):
    new_kitti_vehicle_data = {}
    new_kitti_vehicle_data['kitti_criteria_props'] = {
        'bbox_height': max(vehicle_cav1['kitti_criteria_props']['bbox_height'], vehicle_cav2['kitti_criteria_props']['bbox_height']),
        'occlusion': min(vehicle_cav1['kitti_criteria_props']['occlusion'], vehicle_cav2['kitti_criteria_props']['occlusion']),
        'truncation': min(vehicle_cav1['kitti_criteria_props']['truncation'], vehicle_cav2['kitti_criteria_props']['truncation'])
    }

    # update vehicle_cav1 with kitti criteria of vehicle_cav2 (if the criteria is lower)
    # new_kitti_vehicle_data['kitti_criteria'] = min(vehicle_cav1['kitti_criteria'], vehicle_cav2['kitti_criteria'])

    # calculate the new kitti criteria
    new_kitti_vehicle_data['kitti_criteria'] = set_category_by_camera_props(new_kitti_vehicle_data['kitti_criteria_props'], category_config)

    return new_kitti_vehicle_data
