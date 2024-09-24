"""
Basedataset class for all fusion methods
"""
import os
import tqdm
from opencood.hypes_yaml.yaml_utils import load_yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


# Helper function to process a single cav folder
def process_cav_folder(cav_path, cav_id):
    cav_dict = {}
    timestamps = sorted(os.listdir(cav_path))
    timestamps = [x for x in timestamps if x.endswith('.yaml') and not x.endswith('_additional.yaml')]

    for timestamp in timestamps:
        c_timestamp = timestamp.split('.')[0]
        yaml_content = load_yaml(os.path.join(cav_path, timestamp))
        cav_dict[c_timestamp] = yaml_content

    return cav_id, cav_dict

# Parallelized function to process each folder
def process_folder(folder, folder_path):
    folder_dict = {}
    for cav_id in os.listdir(folder_path):
        if not cav_id.isdigit():
            continue
        cav_path = os.path.join(folder_path, cav_id)
        cav_id = int(cav_id)
        cav_id, cav_data = process_cav_folder(cav_path, cav_id)
        folder_dict[cav_id] = cav_data
    return folder, folder_dict

# Main function to load and store data in parallel
def load_and_store_in_one_file(data_main_path, output_path):
    yaml_dict = {}

    # Collect all folders that need processing
    folders = [folder for folder in os.listdir(data_main_path)
               if os.path.isdir(os.path.join(data_main_path, folder)) and folder != 'additional']

    with ProcessPoolExecutor() as executor:
        futures = []
        for folder in folders:
            folder_path = os.path.join(data_main_path, folder)
            futures.append(executor.submit(process_folder, folder, folder_path))

        # Iterate through completed tasks and store results
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            folder, folder_data = future.result()
            yaml_dict[folder] = folder_data

    # Store the resulting dictionary in a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(yaml_dict, f)


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    data_main_path = r'/data/public_datasets/OPV2V/original'

    train_path = os.path.join(data_main_path, 'train', 'additional')
    load_and_store_in_one_file(train_path, os.path.join(train_path, 'yamls.pkl'))

    validate_path = os.path.join(data_main_path, 'validate', 'additional')
    load_and_store_in_one_file(validate_path, os.path.join(validate_path, 'yamls.pkl'))

    test_path = os.path.join(data_main_path, 'test', 'additional')
    load_and_store_in_one_file(test_path, os.path.join(test_path, 'yamls.pkl'))
