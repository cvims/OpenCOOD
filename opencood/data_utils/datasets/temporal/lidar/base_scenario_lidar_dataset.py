"""
A plain dataset class for lidar
"""
from collections import OrderedDict

import numpy as np

from opencood.utils import box_utils, common_utils, camera_utils
from opencood.data_utils.datasets.temporal.base_scenario_dataset import BaseScenarioDataset
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor

from opencood.data_utils.datasets import COM_RANGE


class BaseScenarioLidarDataset(BaseScenarioDataset):
    def __init__(self, params, visualize, train=True, validate=False, timestamp_offset: int = 0, **kwargs):
        super(BaseScenarioLidarDataset, self).__init__(params, visualize, train,
                                                validate, timestamp_offset=timestamp_offset, **kwargs)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']
    
    def get_sample_random(self, idx):
        return self.retrieve_base_data(idx, self.cur_ego_pose_flag, load_camera_data=False, load_lidar_data=True)


    