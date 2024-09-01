import os
import random
import numpy as np
import tqdm
import opencood.hypes_yaml.yaml_utils as yaml_utils



def check_temporal_potential(ego_cav_id, visibilities, rasterized_range):
    print(f'TODO Checking temporal potential for {ego_cav_id}')
    pass


if __name__ == '__main__':
    dataset_path = r'/data/public_datasets/OPV2V/original/validate/additional'

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
        # {timestamp: {cav_id: {vehicle_id: hits}}}
        visibilities = {}
        for timestamp in timestamps:
            visibilities[timestamp] = {}
            for cav_id in available_cavs:
                cav_path = os.path.join(dataset_path, folder, cav_id)
                cav_yaml_file = os.path.join(cav_path, timestamp)
                yaml_content = yaml_utils.load_yaml(cav_yaml_file)
                visibilities[timestamp][cav_id] = yaml_content['lidar_hits']
        
        temporal_potential_stats[folder] = check_temporal_potential(ego_cav_id, visibilities, rasterized_range)

    print(temporal_potential_stats)
