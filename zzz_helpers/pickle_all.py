# pickle all pickle files into one pickle file
# save the filename as a dict key and the content as a dict value

import os
import pickle as pkl
from tqdm import tqdm


def pickle_all(folder_path, save_path):
    """
    Pickle all pickle files into one pickle file.

    Parameters
    ----------
    folder_path : str
        The folder path.

    save_path : str
        The save path.
    """
    files = os.listdir(folder_path)
    all_data = {}
    for file in tqdm(files):
        with open(os.path.join(folder_path, file), 'rb') as f:
            data = pkl.load(f)
            all_data[file] = data

    with open(save_path, 'wb') as f:
        pkl.dump(all_data, f)


if __name__ == '__main__':
    folder_path = r'visualization/inference/scope'
    save_path = r'visualization/inference/scope/output.pkl'

    pickle_all(folder_path, save_path)
