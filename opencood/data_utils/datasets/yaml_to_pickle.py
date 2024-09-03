"""
Basedataset class for all fusion methods
"""
import os
import tqdm
from opencood.hypes_yaml.yaml_utils import load_yaml
import pickle


def load_and_store_in_one_file(data_main_path, output_path):
    yaml_dict = {}
    # iterate all yamls from all scenarios and all cavs
    for folder in tqdm.tqdm(os.listdir(data_main_path)):
        folder_path = os.path.join(data_main_path, folder)
        # if it is not a folder, skip
        if not os.path.isdir(folder_path):
            continue
        if folder == 'additional':
            continue
        yaml_dict[folder] = {}
        for cav_id in os.listdir(folder_path):
            if not cav_id.isdigit():
                continue
            cav_path = os.path.join(folder_path, cav_id)
            cav_id = int(cav_id)
            yaml_dict[folder][cav_id] = {}
            timestamps = sorted(os.listdir(cav_path))
            timestamps = [x for x in timestamps if x.endswith('.yaml') and not x.endswith('_additional.yaml')]

            for timestamp in timestamps:
                c_timestamp = timestamp.split('.')[0]
                yaml_content = load_yaml(os.path.join(cav_path, timestamp))
                yaml_dict[folder][cav_id][c_timestamp] = yaml_content


    with open(output_path, 'wb') as f:
        pickle.dump(yaml_dict, f)


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    data_main_path = r'/data/public_datasets/OPV2V/original'

    train_path = os.path.join(data_main_path, 'train')
    import time
    start = time.time()
    load_and_store_in_one_file(train_path, os.path.join(train_path, 'yamls.pkl'))
    print('Time taken:', time.time() - start)

    start = time.time()
    train_pkl_file = load_pkl_file(os.path.join(train_path, 'yamls.pkl'))
    print('Time taken:', time.time() - start)

    validate_path = os.path.join(data_main_path, 'validate')
    load_and_store_in_one_file(validate_path, os.path.join(validate_path, 'yamls.pkl'))

    test_path = os.path.join(data_main_path, 'test')
    load_and_store_in_one_file(test_path, os.path.join(test_path, 'yamls.pkl'))
