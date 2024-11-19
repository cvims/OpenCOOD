import os
from torch.utils.data import Dataset
import numpy as np
import tqdm
import pickle


class SCOPEFeatureDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_files = sorted(os.listdir(data_path))

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        with open(os.path.join(self.data_path, self.data_files[idx]), 'rb') as f:
            pkl_data = pickle.load(f)

        feature_list = pkl_data['feature_list']
        feature_2d_list = pkl_data['feature_2d_list']

        object_bbx_centers = pkl_data['object_bbx_centers']
        gt_object_ids_criteria = pkl_data['gt_object_ids_criteria']

        return dict(
            feature_list=feature_list,
            feature_2d_list=feature_2d_list,
            object_bbx_centers=object_bbx_centers,
            gt_object_ids_criteria=gt_object_ids_criteria
        )


if __name__ == '__main__':
    dataset = SCOPEFeatureDataset('temporal_scope_features')
    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]
        print(data)
        break
