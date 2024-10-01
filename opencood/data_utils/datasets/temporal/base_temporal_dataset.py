"""
Basedataset class for lidar data pre-processing
"""
from collections import OrderedDict
import numpy as np

from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.camera_utils import load_rgb_from_files
from opencood.utils.transformation_utils import calculate_prev_pose_offset
from opencood.data_utils.datasets.basedataset import BaseDataset


class BaseTemporalDataset(BaseDataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index. Additionally, it samples multiple frames for each scenario ().

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True, validate=False, **kwargs):       
        if 'queue_length' in params['fusion']['args']:
            self.queue_length = max(1, params['fusion']['args']['queue_length'])
        else:
            self.queue_length = 1
        
        if 'timestamp_offset' in params['fusion']['args']:
            self.timestamp_offset = max(0, params['fusion']['args']['timestamp_offset'])
        else:
            self.timestamp_offset = 0
        
        if 'timestamp_offset_mean' in params['fusion']['args']:
            self.timestamp_offset_mean = params['fusion']['args']['timestamp_offset_mean']
            self.timestamp_offset_std = params['fusion']['args']['timestamp_offset_std']
        else:
            self.timestamp_offset_mean = 0
            self.timestamp_offset_std = 0
        
        self.sensor_cache_container = kwargs.get('sensor_cache_container', None)

        # all previous timestamps are with ego data only (no cooperation)
        # only the current timestamp (last data point) has cooperation
        self.temporal_ego_only = params['fusion']['args']['temporal_ego_only'] if 'temporal_ego_only' in params['fusion']['args'] else False

        super(BaseTemporalDataset, self).__init__(params, visualize, train, validate, **kwargs)


    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        raise NotImplementedError

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int or tuple
            Index given by dataloader or given scenario index and timestamp.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'
        # if the queue length is set to 1, then the vehicle offset is already the connection to the previous car
        # otherwise it starts with the second vehicle (e.g. the first vehicle offset is the identity matrix)

        # we loop the accumulated length list to see get the scenario index
        if isinstance(idx, int):
            scenario_database, timestamp_indices, prev_bev_exists, timestamp_offset = self.retrieve_by_idx(idx)
        else:
            import sys
            sys.exit('Index has to be a int')
        
        ego_cav_content = [cav_content for _, cav_content in scenario_database.items() if cav_content['ego']][0]

        data_queue = []
        # Load files for all timestamps
        for i, (timestamp_index, prev_bev_exist) in enumerate(zip(timestamp_indices, prev_bev_exists)):
            # retrieve the corresponding timestamp key
            timestamp_key = self.return_timestamp_key(
                scenario_database, timestamp_index)

            data = OrderedDict()
            # load files for all CAVs
            for cav_id, cav_content in scenario_database.items():
                if timestamp_key not in cav_content:
                    continue

                if self.temporal_ego_only and i < self.queue_length - 1:
                    if not cav_content['ego']:
                        continue

                data[cav_id] = OrderedDict()
                data[cav_id]['ego'] = cav_content['ego']
                data[cav_id]['vehicles'] = cav_content[timestamp_key]['yaml']['vehicles']
                data[cav_id]['true_ego_pos'] = cav_content[timestamp_key]['yaml']['true_ego_pos']

                # calculate delay for this vehicle
                timestamp_delay = \
                    self.time_delay_calculation(cav_content['ego'])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index

                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                timestamp_index_delay)
                # add time delay to vehicle parameters
                data[cav_id]['time_delay'] = timestamp_delay
                data[cav_id]['timestamp_key'] = timestamp_key_delay

                # load the lidar params into the dictionary
                data[cav_id]['params'] = self.reform_lidar_param(
                    cav_content, ego_cav_content,
                    timestamp_key, timestamp_key_delay, cur_ego_pose_flag)

                if load_lidar_data:
                    data[cav_id]['lidar_np'] = \
                        pcd_to_np(cav_content[timestamp_key_delay]['lidar'], self.sensor_cache_container)

                if load_camera_data:
                    # load the camera transformation matrix to dictionary
                    data[cav_id]['camera_params'] = \
                        self.reform_camera_param(cav_content, ego_cav_content, timestamp_key)
                    data[cav_id]['camera_np'] = \
                        load_rgb_from_files(cav_content[timestamp_key_delay]['cameras'], self.sensor_cache_container)                    

                # # add previous bev information
                # if i == 0:
                #     data[cav_id]['prev_bev_exists'] = False  # Could also be True (see prev_bev_exists), but we keep the first always False
                #     data[cav_id]['prev_pose_offset'] = np.eye(4)
                #     # if queue length is set to 1, use the previous frame (if possible) to calculate the transformation matrix
                #     if self.queue_length == 1:
                #         # offset between current and previous frame
                #         # -1 because we want the previous frame and - timestamp offset (which is the skips in between the frames)
                #         prev_timestamp_index = max(0, timestamp_index_delay - timestamp_offset - 1)
                #         prev_timestamp_index_key = self.return_timestamp_key(scenario_database, prev_timestamp_index)

                #         if prev_timestamp_index >= timestamp_index_delay:
                #             data[cav_id]['prev_bev_exists'] = False
                #             data[cav_id]['prev_pose_offset'] = np.eye(4)
                #         else:
                #             prev_cav_data = dict()
                #             prev_cav_data['params'] = self.reform_lidar_param(
                #                 cav_content, ego_cav_content,
                #                 prev_timestamp_index_key, prev_timestamp_index_key, cur_ego_pose_flag
                #             )
                #             data[cav_id]['prev_bev_exists'] = True
                #             data[cav_id]['prev_pose_offset'] = calculate_prev_pose_offset(data[cav_id], prev_cav_data)
                # else:
                #     data[cav_id]['prev_bev_exists'] = prev_bev_exist
                #     data[cav_id]['prev_pose_offset'] = calculate_prev_pose_offset(
                #         data[cav_id], data_queue[i - 1][cav_id])

            data_queue.append(data)

        return data_queue

    @staticmethod
    def find_ego_pose(base_data_dict):
        """
        Find the ego vehicle id and corresponding LiDAR pose from all cavs.

        Parameters
        ----------
        base_data_dict : dict
            The dictionary contains all basic information of all cavs.

        Returns
        -------
        ego vehicle id and the corresponding lidar pose.
        """

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose


    def retrieve_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.
            We use the query length to get the previous (or subsequent) timestamps

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                _len_record = self.len_record[i - 1] if i > 0 else 0
                scenario_index = i
                timestamp_index = idx if i == 0 else idx - _len_record
                max_idx = self.len_record[i] - _len_record - 1
                break
        
        if self.timestamp_offset_mean > 0 and self.timestamp_offset_std > 0:
            timestamp_offset = max(0, int(np.random.normal(self.timestamp_offset_mean, self.timestamp_offset_std)))
        else:
            timestamp_offset = self.timestamp_offset
        
        # keep the timestamp indices between min_idx and max_idx
        span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        if span > max_idx:
            timestamp_index = 0
            timestamp_offset = (max_idx - self.queue_length) // (self.queue_length - 1)
            span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        # check if its in between min and max idx
        if span > 1:
            if (timestamp_index + span) > max_idx:
                timestamp_index = max(0, timestamp_index - span)

        scenario_database = self.scenario_database[scenario_index]

        timestamp_indices = np.array(
            [timestamp_index + i * (timestamp_offset + 1) for i in range(self.queue_length)]
        )
        
        prev_bevs_exists = []
        if timestamp_index == 0:
            prev_bevs_exists.append(False)
        else:
            prev_bevs_exists.append(True)
        
        for i in range(self.queue_length - 1):
            prev_bevs_exists.append(True)

        return scenario_database, timestamp_indices, prev_bevs_exists, timestamp_offset

    def augment(self, **kwargs):
        """
        """
        raise NotImplementedError

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        raise NotImplementedError
