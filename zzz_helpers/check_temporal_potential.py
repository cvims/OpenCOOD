import os
import yaml
import tqdm


def check_vehicles_in_range(ego_id, vehicles, rasterized_range=50):
    """
    Checks if the vehicles are in the rasterized range of the ego vehicle (bev / quadratic)
    :param ego_id: ego vehicle id
    :param vehicles: list of vehicles
    :param rasterized_range: range to check
    :return: list of vehicle ids in range
    """
    # get ego location
    ego = vehicles[ego_id]
    others = {k: v for k, v in vehicles.items() if k != ego_id}

    ego_loc = ego['location']

    # check if vehicles are in range
    other_in_range_ids = []
    for other_id, other_data in others.items():
        other_loc = other_data['location']
        if abs(ego_loc[0] - other_loc[0]) < rasterized_range and abs(ego_loc[1] - other_loc[1]) < rasterized_range:
            other_in_range_ids.append(other_id)
    
    return other_in_range_ids



def check_temporal_potential(frame_vehicle_ids: list):
    """
    temporal potential exists if a vehicle is visible in at least two frames (except first)
    :param frame_vehicle_ids: list of vehicle ids in each frame
    """
    assert len(frame_vehicle_ids) >= 3, 'At least three frames are required to check temporal potential'

    # last frame is prediction frame (does not count)

    # count id occurences
    id_occurences = {}
    for frame in frame_vehicle_ids[:-1]:
        for id in frame:
            if id not in id_occurences:
                id_occurences[id] = 1
            else:
                id_occurences[id] += 1
    
    id_occurences_last = {}
    for id in frame_vehicle_ids[-1]:
        id_occurences_last[id] = 1

    # check if id is in at least two frames but not in the last

    temporal_potential_count = 0
    for id, occurences in id_occurences.items():
        if occurences >= 2 and id not in id_occurences_last:
            # print(f'Temporal potential for id {id}')
            temporal_potential_count += 1
    
    return temporal_potential_count

if __name__ == '__main__':
    dataset_path = r'/data/public_datasets/OPV2V/original/train_test/additional'
    org_dataset_path = r'/data/public_datasets/OPV2V/original/train'

    frames_to_check = 4
    rasterized_range = 50

    # - timestamp (folder)
    #   - {timestamp}_all_agents.yaml

    folders = os.listdir(dataset_path)

    temporal_potential_count_all = 0
    all_frames_count = 0
    for folder in tqdm.tqdm(folders):
        random_cav_id = [x for x in os.listdir(os.path.join(org_dataset_path, folder)) if x.isdigit()][0]
        full_path = os.path.join(dataset_path, folder)
        # sort by timestamp (str)
        timestamps = sorted(os.listdir(full_path))

        all_frames_count += len(timestamps)

        # main loop, step size 1, shift 1 until end
        for i in range(len(timestamps) - frames_to_check):
            # load yamls
            yaml_files = [timestamps[j] for j in range(i, i + frames_to_check)]
            yaml_content = []
            for yaml_file in yaml_files:
                with open(os.path.join(full_path, yaml_file), 'r') as f:
                    yaml_content.append(yaml.load(f, Loader=yaml.FullLoader))

            # check if vehicles are in range
            in_range_ids = []
            for frame in yaml_content:
                ids = check_vehicles_in_range(random_cav_id, frame['vehicles'], rasterized_range)
                in_range_ids.append(ids)
            
            # check temporal potential
            temporal_potential_count_all += check_temporal_potential(in_range_ids)
    
    print(f'All frames count: {all_frames_count}')
    print(f'Temporal potential count: {temporal_potential_count_all}')
