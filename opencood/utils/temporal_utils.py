VISIBLITY_CATEGORY_ENUM = {
    'easy': 0,
    'moderate': 1,
    'hard': 2,
    'very_hard': 3,
    'none': 4
}


def set_category_by_lidar_hits(lidar_hits: int, category_config: dict):
    easy = category_config['easy']['hits']
    moderate = category_config['moderate']['hits']
    hard = category_config['hard']['hits']
    very_hard = category_config['very_hard']['hits']
    none = category_config['none']['hits']

    if lidar_hits >= easy:
        return VISIBLITY_CATEGORY_ENUM['easy']
    elif lidar_hits >= moderate:
        return VISIBLITY_CATEGORY_ENUM['moderate']
    elif lidar_hits >= hard:
        return VISIBLITY_CATEGORY_ENUM['hard']
    elif lidar_hits >= very_hard:
        return VISIBLITY_CATEGORY_ENUM['very_hard']
    else:
        return VISIBLITY_CATEGORY_ENUM['none']


def set_category_by_camera_props(camera_props: dict, category_config: dict):
    # TODO
    print('Camera props are not yet implemented.')

    easy = category_config['easy']
    moderate = category_config['moderate']
    hard = category_config['hard']['args']
    very_hard = category_config['very_hard']
    none = category_config['none']

    if camera_props['occlusion'] <= easy['occlusion']:
        return VISIBLITY_CATEGORY_ENUM['easy']
    elif camera_props['occlusion'] <= moderate['occlusion']:
        return VISIBLITY_CATEGORY_ENUM['moderate']
    elif camera_props['occlusion'] <= hard['occlusion']:
        return VISIBLITY_CATEGORY_ENUM['hard']
    elif camera_props['occlusion'] <= very_hard['occlusion']:
        return VISIBLITY_CATEGORY_ENUM['very_hard']
    else:
        return VISIBLITY_CATEGORY_ENUM['none']
    

def categorize_vehicle_visibility_by_lidar_hits(vehicle_list: list, category_config: dict):
    """
    Categorize vehicles in the temporal vehicles list by the lidar hits.
    """

    for vehicle in vehicle_list.values():
        vehicle['lidar_visibility'] = set_category_by_lidar_hits(vehicle['lidar_hits'], category_config)
    
    return vehicle_list


def categorize_vehicle_visibility_by_camera_props(vehicle_list: list, category_config: dict):
    """
    Categorize vehicles in the temporal vehicles list by the camera properties.
    """
    for vehicle in vehicle_list.values():
        # vehicle['camera_visibility'] = set_category_by_camera_props(vehicle['camera_props'], category_config)
        vehicle['camera_visibility'] = VISIBLITY_CATEGORY_ENUM['moderate']
    
    return vehicle_list


def is_lower_or_equal_visibility_category(vehicle_category: int, min_category: int):
    """
    Check if the vehicle category is lower or equal to the minimum category.
    """
    return vehicle_category <= min_category


def filter_vehicles_by_category(vehicles: dict, min_category: int, is_camera: bool):
    """
    Filter vehicles by category.
    :param vehicles: dict of vehicles
    :param min_category: minimum category, e.g. 'hard' means that only vehicles with category 'hard' or lower will be returned
    """
    if is_camera:
        category_key = 'camera_visibility'
    else:
        category_key = 'lidar_visibility'

    filtered_entry = {}
    for v_id, vehicle in vehicles.items():
        if is_lower_or_equal_visibility_category(vehicle[category_key], min_category):
            filtered_entry[v_id] = vehicle

    return filtered_entry


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
        vehicle['temporal_min_lidar_visibility'] = vehicle['lidar_visibility']
        vehicle['temporal_min_camera_visibility'] = vehicle['camera_visibility']

    for i in range(1, len(temporal_vehicles_list)):
        current_frame = temporal_vehicles_list[i]
        prev_frame = updated_temporal_vehicles_list[-1]

        for vehicle in current_frame.values():
            vehicle['temporal_min_lidar_visibility'] = vehicle['lidar_visibility']
            vehicle['temporal_min_camera_visibility'] = vehicle['camera_visibility']

        current_frame_ids = list(current_frame.keys())
        prev_frame_ids = list(prev_frame.keys())

        for v_id in prev_frame_ids:
            if v_id not in current_frame_ids and v_id in all_in_range_vehicles_list[i]:
                # add the vehicle to the current frame if it was visible in the previous frame (with the coordinates of the current frame)
                current_frame[v_id] = all_in_range_vehicles_list[i][v_id]
                # Preserve the visibility data from the previous frame
                current_frame[v_id]['lidar_visibility'] = prev_frame[v_id].get('lidar_visibility')
                current_frame[v_id]['camera_visibility'] = prev_frame[v_id].get('camera_visibility')
        
        # update visibility category. Always the lowest category in the temporal stream
        for vehicle in current_frame.values():
            # Update lidar visibility if applicable
            vehicle['temporal_min_lidar_visibility'] = min(
                vehicle.get('temporal_min_lidar_visibility', vehicle['lidar_visibility']),
                vehicle['lidar_visibility']
            )
            # Update camera visibility if applicable
            vehicle['temporal_min_camera_visibility'] = min(
                vehicle.get('temporal_min_camera_visibility', vehicle['camera_visibility']),
                vehicle['camera_visibility']
            )
        
        updated_temporal_vehicles_list.append(current_frame)
    
    return updated_temporal_vehicles_list
