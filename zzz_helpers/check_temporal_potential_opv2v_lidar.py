import os
import random
import numpy as np
import tqdm
import opencood.hypes_yaml.yaml_utils as yaml_utils



def check_temporal_potential(ego_cav_id, cav_poses, visibilities, frames_to_check, rasterized_range, min_points_threshold=10):
    # frames to check is the amount of frames that are considered for temporal potential
    # e.g. frames_to_check=4 means that the current frame and the next 3 frames are considered
    # Conditions for temporal potential:
    #   1. The vehicle must be at least one time invisible in the frames_to_check
    #   2. We need at least 2 frames were the vehicle is visible
    # Possible examples of temporal potential:
    #   1. Frame 1: visible, Frame 2: visible, Frame 3: visible, Frame 4: invisible
    #   2. Frame 1: visible, Frame 2: invisible, Frame 3: visible, Frame 4: invisible
    #   3. Frame 1: visible, Frame 2: visible, Frame 3: invisible, Frame 4: invisible

    # iterate through timestamps
    temporal_potential_stats = {}

    frames_stats = {}
    for timestamp, cav_data in visibilities.items():
        frames_stats[timestamp] = {}

        ego_pose = cav_poses[ego_cav_id][timestamp]
        ego_location = np.asarray(ego_pose[:3])

        in_range_ids = set()
        for v_id, v_content in cav_data[ego_cav_id].items():
            location = np.asarray(v_content['location'])

            # check if vehicle is in range of ego considering rasterized range
            if np.linalg.norm(location - ego_location) > rasterized_range:
                continue

            in_range_ids.add(v_id)

        for v_data in cav_data.values():
            for v_id, v_content in v_data.items():
                if v_id not in in_range_ids:
                    continue

                if v_id not in frames_stats[timestamp]:
                    frames_stats[timestamp][v_id] = 0

                frames_stats[timestamp][v_id] += v_content['lidar_hits']
    
    # analyse potential (t -> t+frames_to_check, t+1 -> t+1+frames_to_check, ...)
    for i, timestamp in enumerate(frames_stats.keys()):
        if i + frames_to_check >= len(frames_stats):
            break

        # check if vehicle is visible in the next frames_to_check frames
        visibility_count = {}
        for j in range(frames_to_check):
            next_timestamp = list(frames_stats.keys())[list(frames_stats.keys()).index(timestamp) + j]
            for v_id in frames_stats[next_timestamp]:
                if frames_stats[next_timestamp][v_id] >= min_points_threshold:
                    if v_id not in visibility_count:
                        visibility_count[v_id] = 0
                    visibility_count[v_id] += 1
        
        # all entries > 2 and smaller than frames_to_check are considered as temporal potential
        for v_id, count in visibility_count.items():
            if count >= 2 and count < frames_to_check:
                if timestamp not in temporal_potential_stats:
                    temporal_potential_stats[timestamp] = []
                temporal_potential_stats[timestamp].append(v_id)
        
    return temporal_potential_stats

if __name__ == '__main__':
    # seed
    random.seed(42)
    dataset_path = r'/data/public_datasets/OPV2V/original/train/additional'

    ## CONFIG ##
    frames_to_check = 4
    rasterized_range = 100  # 100m for lidar

    max_cavs = 1
    ############

    folders = os.listdir(dataset_path)

    temporal_potential_stats = {}
    for folder in tqdm.tqdm(folders):
        available_cavs = [x for x in os.listdir(os.path.join(dataset_path, folder)) if x.isdigit()]
        # random shuffle available cavs
        random.shuffle(available_cavs)

        if len(available_cavs) > max_cavs:
            available_cavs = available_cavs[:max_cavs]
        
        # choose first cav as ego
        ego_cav_id = available_cavs[0] # this is for reference
        # load cav timestamps (ego)
        full_path = os.path.join(dataset_path, folder, ego_cav_id)
        # sort by timestamp (str)
        timestamps = sorted(os.listdir(full_path))
        # filter {timestamp}.yaml
        timestamps = [x for x in timestamps if x.endswith('.yaml') and not x.endswith('_additional.yaml')]

        # save vehicle ids and their lidar hits, hits are the sum of hits across all cavs
        visibilities = {}
        cav_poses = {cav_id: {} for cav_id in available_cavs}
        for timestamp_yaml in timestamps:
            timestamp = timestamp_yaml.split('.')[0]
            visibilities[timestamp] = {}
            for cav_id in available_cavs:
                visibilities[timestamp][cav_id] = {}
                cav_path = os.path.join(dataset_path, folder, cav_id)
                cav_yaml_file = os.path.join(cav_path, timestamp_yaml)
                yaml_content = yaml_utils.load_yaml(cav_yaml_file)
                vehicles = yaml_content['vehicles']
                for v_id, v_data in vehicles.items():
                    visibilities[timestamp][cav_id][v_id] = v_data
                # add pose
                cav_poses[cav_id][timestamp] = yaml_content['true_ego_pos']
        
        temporal_potential_stats[folder] = check_temporal_potential(ego_cav_id, cav_poses, visibilities, frames_to_check, rasterized_range)

    print(temporal_potential_stats)
